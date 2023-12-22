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
    x_1 = torch.nn.functional.batch_norm(x, l__mod___bn1_running_mean, l__mod___bn1_running_var, l__mod___bn1_weight, l__mod___bn1_bias, False, 0.1, 0.001);  x = l__mod___bn1_running_mean = l__mod___bn1_running_var = l__mod___bn1_weight = l__mod___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___bn1_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___bn1_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_5 = self.getattr_getattr_L__mod___blocks___0_____0___conv_pw(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn1_running_var, getattr_getattr_l__mod___blocks___0_____0___bn1_weight, getattr_getattr_l__mod___blocks___0_____0___bn1_bias, False, 0.1, 0.001);  x_5 = getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = getattr_getattr_l__mod___blocks___0_____0___bn1_weight = getattr_getattr_l__mod___blocks___0_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_10 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_11 = torch.nn.functional.batch_norm(x_10, getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn2_running_var, getattr_getattr_l__mod___blocks___0_____0___bn2_weight, getattr_getattr_l__mod___blocks___0_____0___bn2_bias, False, 0.1, 0.001);  x_10 = getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = getattr_getattr_l__mod___blocks___0_____0___bn2_weight = getattr_getattr_l__mod___blocks___0_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_12 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_14 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_15 = self.getattr_getattr_L__mod___blocks___0_____0___se(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_16 = self.getattr_getattr_L__mod___blocks___0_____0___conv_pwl(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___0_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_16, getattr_getattr_l__mod___blocks___0_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn3_running_var, getattr_getattr_l__mod___blocks___0_____0___bn3_weight, getattr_getattr_l__mod___blocks___0_____0___bn3_bias, False, 0.1, 0.001);  x_16 = getattr_getattr_l__mod___blocks___0_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn3_running_var = getattr_getattr_l__mod___blocks___0_____0___bn3_weight = getattr_getattr_l__mod___blocks___0_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.getattr_getattr_L__mod___blocks___0_____0___bn3_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_20 = self.getattr_getattr_L__mod___blocks___0_____0___bn3_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___0_____0___drop_path = self.getattr_getattr_L__mod___blocks___0_____0___drop_path(x_20);  x_20 = None
    shortcut_1 = getattr_getattr_l__mod___blocks___0_____0___drop_path + shortcut;  getattr_getattr_l__mod___blocks___0_____0___drop_path = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_22 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_23 = torch.nn.functional.batch_norm(x_22, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, False, 0.1, 0.001);  x_22 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_24 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_26 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_27 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_28 = torch.nn.functional.batch_norm(x_27, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, False, 0.1, 0.001);  x_27 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_29 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_31 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_32 = self.getattr_getattr_L__mod___blocks___1_____0___se(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_33 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_34 = torch.nn.functional.batch_norm(x_33, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, False, 0.1, 0.001);  x_33 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_35 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_38 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, False, 0.1, 0.001);  x_38 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_42 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_43 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_44 = torch.nn.functional.batch_norm(x_43, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, False, 0.1, 0.001);  x_43 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_45 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_47 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_48 = self.getattr_getattr_L__mod___blocks___1_____1___se(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_49 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_50 = torch.nn.functional.batch_norm(x_49, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, False, 0.1, 0.001);  x_49 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_51 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_53 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_53);  x_53 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_55 = self.getattr_getattr_L__mod___blocks___1_____2___conv_pw(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_56 = torch.nn.functional.batch_norm(x_55, getattr_getattr_l__mod___blocks___1_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____2___bn1_running_var, getattr_getattr_l__mod___blocks___1_____2___bn1_weight, getattr_getattr_l__mod___blocks___1_____2___bn1_bias, False, 0.1, 0.001);  x_55 = getattr_getattr_l__mod___blocks___1_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____2___bn1_running_var = getattr_getattr_l__mod___blocks___1_____2___bn1_weight = getattr_getattr_l__mod___blocks___1_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_57 = self.getattr_getattr_L__mod___blocks___1_____2___bn1_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_59 = self.getattr_getattr_L__mod___blocks___1_____2___bn1_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_60 = self.getattr_getattr_L__mod___blocks___1_____2___conv_dw(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_61 = torch.nn.functional.batch_norm(x_60, getattr_getattr_l__mod___blocks___1_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____2___bn2_running_var, getattr_getattr_l__mod___blocks___1_____2___bn2_weight, getattr_getattr_l__mod___blocks___1_____2___bn2_bias, False, 0.1, 0.001);  x_60 = getattr_getattr_l__mod___blocks___1_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____2___bn2_running_var = getattr_getattr_l__mod___blocks___1_____2___bn2_weight = getattr_getattr_l__mod___blocks___1_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_62 = self.getattr_getattr_L__mod___blocks___1_____2___bn2_drop(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_64 = self.getattr_getattr_L__mod___blocks___1_____2___bn2_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_65 = self.getattr_getattr_L__mod___blocks___1_____2___se(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_66 = self.getattr_getattr_L__mod___blocks___1_____2___conv_pwl(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_67 = torch.nn.functional.batch_norm(x_66, getattr_getattr_l__mod___blocks___1_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____2___bn3_running_var, getattr_getattr_l__mod___blocks___1_____2___bn3_weight, getattr_getattr_l__mod___blocks___1_____2___bn3_bias, False, 0.1, 0.001);  x_66 = getattr_getattr_l__mod___blocks___1_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____2___bn3_running_var = getattr_getattr_l__mod___blocks___1_____2___bn3_weight = getattr_getattr_l__mod___blocks___1_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_68 = self.getattr_getattr_L__mod___blocks___1_____2___bn3_drop(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_70 = self.getattr_getattr_L__mod___blocks___1_____2___bn3_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____2___drop_path = self.getattr_getattr_L__mod___blocks___1_____2___drop_path(x_70);  x_70 = None
    shortcut_4 = getattr_getattr_l__mod___blocks___1_____2___drop_path + shortcut_3;  getattr_getattr_l__mod___blocks___1_____2___drop_path = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_72 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_4);  shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_73 = torch.nn.functional.batch_norm(x_72, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, False, 0.1, 0.001);  x_72 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_74 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_76 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_77 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_78 = torch.nn.functional.batch_norm(x_77, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, False, 0.1, 0.001);  x_77 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_79 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_81 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_82 = self.getattr_getattr_L__mod___blocks___2_____0___se(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_83 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_84 = torch.nn.functional.batch_norm(x_83, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, False, 0.1, 0.001);  x_83 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_85 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_5 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_88 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_88, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, False, 0.1, 0.001);  x_88 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_92 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_93 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_94 = torch.nn.functional.batch_norm(x_93, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, False, 0.1, 0.001);  x_93 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_95 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_97 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_98 = self.getattr_getattr_L__mod___blocks___2_____1___se(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_99 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_100 = torch.nn.functional.batch_norm(x_99, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, False, 0.1, 0.001);  x_99 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_101 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_103 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_103);  x_103 = None
    shortcut_6 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_5;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_105 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn1_running_var, getattr_getattr_l__mod___blocks___2_____2___bn1_weight, getattr_getattr_l__mod___blocks___2_____2___bn1_bias, False, 0.1, 0.001);  x_105 = getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = getattr_getattr_l__mod___blocks___2_____2___bn1_weight = getattr_getattr_l__mod___blocks___2_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_109 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_110 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_111 = torch.nn.functional.batch_norm(x_110, getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn2_running_var, getattr_getattr_l__mod___blocks___2_____2___bn2_weight, getattr_getattr_l__mod___blocks___2_____2___bn2_bias, False, 0.1, 0.001);  x_110 = getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = getattr_getattr_l__mod___blocks___2_____2___bn2_weight = getattr_getattr_l__mod___blocks___2_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_112 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_114 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_act(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_115 = self.getattr_getattr_L__mod___blocks___2_____2___se(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_116 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_117 = torch.nn.functional.batch_norm(x_116, getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn3_running_var, getattr_getattr_l__mod___blocks___2_____2___bn3_weight, getattr_getattr_l__mod___blocks___2_____2___bn3_bias, False, 0.1, 0.001);  x_116 = getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = getattr_getattr_l__mod___blocks___2_____2___bn3_weight = getattr_getattr_l__mod___blocks___2_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_118 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_120 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____2___drop_path = self.getattr_getattr_L__mod___blocks___2_____2___drop_path(x_120);  x_120 = None
    shortcut_7 = getattr_getattr_l__mod___blocks___2_____2___drop_path + shortcut_6;  getattr_getattr_l__mod___blocks___2_____2___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_122 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pw(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_123 = torch.nn.functional.batch_norm(x_122, getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn1_running_var, getattr_getattr_l__mod___blocks___2_____3___bn1_weight, getattr_getattr_l__mod___blocks___2_____3___bn1_bias, False, 0.1, 0.001);  x_122 = getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn1_running_var = getattr_getattr_l__mod___blocks___2_____3___bn1_weight = getattr_getattr_l__mod___blocks___2_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_124 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_drop(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_126 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_act(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_127 = self.getattr_getattr_L__mod___blocks___2_____3___conv_dw(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_128 = torch.nn.functional.batch_norm(x_127, getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn2_running_var, getattr_getattr_l__mod___blocks___2_____3___bn2_weight, getattr_getattr_l__mod___blocks___2_____3___bn2_bias, False, 0.1, 0.001);  x_127 = getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn2_running_var = getattr_getattr_l__mod___blocks___2_____3___bn2_weight = getattr_getattr_l__mod___blocks___2_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_129 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_131 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_132 = self.getattr_getattr_L__mod___blocks___2_____3___se(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_133 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pwl(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn3_running_var, getattr_getattr_l__mod___blocks___2_____3___bn3_weight, getattr_getattr_l__mod___blocks___2_____3___bn3_bias, False, 0.1, 0.001);  x_133 = getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn3_running_var = getattr_getattr_l__mod___blocks___2_____3___bn3_weight = getattr_getattr_l__mod___blocks___2_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_137 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____3___drop_path = self.getattr_getattr_L__mod___blocks___2_____3___drop_path(x_137);  x_137 = None
    shortcut_8 = getattr_getattr_l__mod___blocks___2_____3___drop_path + shortcut_7;  getattr_getattr_l__mod___blocks___2_____3___drop_path = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_139 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_8);  shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_139, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, False, 0.1, 0.001);  x_139 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_143 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_144 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_145 = torch.nn.functional.batch_norm(x_144, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, False, 0.1, 0.001);  x_144 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_146 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_148 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_149 = self.getattr_getattr_L__mod___blocks___3_____0___se(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_150 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_151 = torch.nn.functional.batch_norm(x_150, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, False, 0.1, 0.001);  x_150 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_9 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_155 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_156 = torch.nn.functional.batch_norm(x_155, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, False, 0.1, 0.001);  x_155 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_157 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_159 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_160 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_161 = torch.nn.functional.batch_norm(x_160, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, False, 0.1, 0.001);  x_160 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_162 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_164 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_165 = self.getattr_getattr_L__mod___blocks___3_____1___se(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_166 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_167 = torch.nn.functional.batch_norm(x_166, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, False, 0.1, 0.001);  x_166 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_168 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_170 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_170);  x_170 = None
    shortcut_10 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_9;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_172 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_172, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, False, 0.1, 0.001);  x_172 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_176 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_177 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_178 = torch.nn.functional.batch_norm(x_177, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, False, 0.1, 0.001);  x_177 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_179 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_181 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_182 = self.getattr_getattr_L__mod___blocks___3_____2___se(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_183 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_184 = torch.nn.functional.batch_norm(x_183, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, False, 0.1, 0.001);  x_183 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_185 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_187);  x_187 = None
    shortcut_11 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_10;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_189 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_190 = torch.nn.functional.batch_norm(x_189, getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn1_running_var, getattr_getattr_l__mod___blocks___3_____3___bn1_weight, getattr_getattr_l__mod___blocks___3_____3___bn1_bias, False, 0.1, 0.001);  x_189 = getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = getattr_getattr_l__mod___blocks___3_____3___bn1_weight = getattr_getattr_l__mod___blocks___3_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_191 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_193 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_act(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_194 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_195 = torch.nn.functional.batch_norm(x_194, getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn2_running_var, getattr_getattr_l__mod___blocks___3_____3___bn2_weight, getattr_getattr_l__mod___blocks___3_____3___bn2_bias, False, 0.1, 0.001);  x_194 = getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = getattr_getattr_l__mod___blocks___3_____3___bn2_weight = getattr_getattr_l__mod___blocks___3_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_196 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_drop(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_198 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_act(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_199 = self.getattr_getattr_L__mod___blocks___3_____3___se(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_200 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_201 = torch.nn.functional.batch_norm(x_200, getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn3_running_var, getattr_getattr_l__mod___blocks___3_____3___bn3_weight, getattr_getattr_l__mod___blocks___3_____3___bn3_bias, False, 0.1, 0.001);  x_200 = getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = getattr_getattr_l__mod___blocks___3_____3___bn3_weight = getattr_getattr_l__mod___blocks___3_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_202 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_drop(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_204 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_act(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____3___drop_path = self.getattr_getattr_L__mod___blocks___3_____3___drop_path(x_204);  x_204 = None
    shortcut_12 = getattr_getattr_l__mod___blocks___3_____3___drop_path + shortcut_11;  getattr_getattr_l__mod___blocks___3_____3___drop_path = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_206 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_12);  shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_207 = torch.nn.functional.batch_norm(x_206, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, False, 0.1, 0.001);  x_206 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_208 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_210 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_211 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_212 = torch.nn.functional.batch_norm(x_211, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, False, 0.1, 0.001);  x_211 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_213 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_215 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_216 = self.getattr_getattr_L__mod___blocks___4_____0___se(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_217 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_218 = torch.nn.functional.batch_norm(x_217, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, False, 0.1, 0.001);  x_217 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_219 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_13 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_222 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_223 = torch.nn.functional.batch_norm(x_222, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, False, 0.1, 0.001);  x_222 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_224 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_226 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_227 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, False, 0.1, 0.001);  x_227 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_231 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_232 = self.getattr_getattr_L__mod___blocks___4_____1___se(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_233 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_234 = torch.nn.functional.batch_norm(x_233, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, False, 0.1, 0.001);  x_233 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_235 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_237 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_237);  x_237 = None
    shortcut_14 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_13;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_239 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_240 = torch.nn.functional.batch_norm(x_239, getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn1_running_var, getattr_getattr_l__mod___blocks___4_____2___bn1_weight, getattr_getattr_l__mod___blocks___4_____2___bn1_bias, False, 0.1, 0.001);  x_239 = getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = getattr_getattr_l__mod___blocks___4_____2___bn1_weight = getattr_getattr_l__mod___blocks___4_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_241 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_drop(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_243 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_act(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_244 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_245 = torch.nn.functional.batch_norm(x_244, getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn2_running_var, getattr_getattr_l__mod___blocks___4_____2___bn2_weight, getattr_getattr_l__mod___blocks___4_____2___bn2_bias, False, 0.1, 0.001);  x_244 = getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = getattr_getattr_l__mod___blocks___4_____2___bn2_weight = getattr_getattr_l__mod___blocks___4_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_246 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_drop(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_248 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_act(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_249 = self.getattr_getattr_L__mod___blocks___4_____2___se(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_250 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_251 = torch.nn.functional.batch_norm(x_250, getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn3_running_var, getattr_getattr_l__mod___blocks___4_____2___bn3_weight, getattr_getattr_l__mod___blocks___4_____2___bn3_bias, False, 0.1, 0.001);  x_250 = getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = getattr_getattr_l__mod___blocks___4_____2___bn3_weight = getattr_getattr_l__mod___blocks___4_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_252 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_drop(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_254 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____2___drop_path = self.getattr_getattr_L__mod___blocks___4_____2___drop_path(x_254);  x_254 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___4_____2___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___4_____2___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_256 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_257 = torch.nn.functional.batch_norm(x_256, getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn1_running_var, getattr_getattr_l__mod___blocks___4_____3___bn1_weight, getattr_getattr_l__mod___blocks___4_____3___bn1_bias, False, 0.1, 0.001);  x_256 = getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = getattr_getattr_l__mod___blocks___4_____3___bn1_weight = getattr_getattr_l__mod___blocks___4_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_258 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_drop(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_260 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_261 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_262 = torch.nn.functional.batch_norm(x_261, getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn2_running_var, getattr_getattr_l__mod___blocks___4_____3___bn2_weight, getattr_getattr_l__mod___blocks___4_____3___bn2_bias, False, 0.1, 0.001);  x_261 = getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = getattr_getattr_l__mod___blocks___4_____3___bn2_weight = getattr_getattr_l__mod___blocks___4_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_263 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_drop(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_265 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_266 = self.getattr_getattr_L__mod___blocks___4_____3___se(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_267 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_268 = torch.nn.functional.batch_norm(x_267, getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn3_running_var, getattr_getattr_l__mod___blocks___4_____3___bn3_weight, getattr_getattr_l__mod___blocks___4_____3___bn3_bias, False, 0.1, 0.001);  x_267 = getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = getattr_getattr_l__mod___blocks___4_____3___bn3_weight = getattr_getattr_l__mod___blocks___4_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_269 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_drop(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_271 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_act(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____3___drop_path = self.getattr_getattr_L__mod___blocks___4_____3___drop_path(x_271);  x_271 = None
    shortcut_16 = getattr_getattr_l__mod___blocks___4_____3___drop_path + shortcut_15;  getattr_getattr_l__mod___blocks___4_____3___drop_path = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_273 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_16);  shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_274 = torch.nn.functional.batch_norm(x_273, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, False, 0.1, 0.001);  x_273 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_275 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_277 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_278 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, False, 0.1, 0.001);  x_278 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_282 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_283 = self.getattr_getattr_L__mod___blocks___5_____0___se(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_284 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_285 = torch.nn.functional.batch_norm(x_284, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, False, 0.1, 0.001);  x_284 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_286 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_17 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_289 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_290 = torch.nn.functional.batch_norm(x_289, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, False, 0.1, 0.001);  x_289 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_291 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_293 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_294 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_295 = torch.nn.functional.batch_norm(x_294, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, False, 0.1, 0.001);  x_294 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_296 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_298 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_299 = self.getattr_getattr_L__mod___blocks___5_____1___se(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_300 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_301 = torch.nn.functional.batch_norm(x_300, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, False, 0.1, 0.001);  x_300 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_302 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_304 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_304);  x_304 = None
    shortcut_18 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_17;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_306 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_307 = torch.nn.functional.batch_norm(x_306, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, False, 0.1, 0.001);  x_306 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_308 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_310 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_311 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_312 = torch.nn.functional.batch_norm(x_311, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, False, 0.1, 0.001);  x_311 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_313 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_315 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_316 = self.getattr_getattr_L__mod___blocks___5_____2___se(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_317 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_318 = torch.nn.functional.batch_norm(x_317, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, False, 0.1, 0.001);  x_317 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_319 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_321 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_321);  x_321 = None
    shortcut_19 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_18;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_323 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pw(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_324 = torch.nn.functional.batch_norm(x_323, getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn1_running_var, getattr_getattr_l__mod___blocks___5_____3___bn1_weight, getattr_getattr_l__mod___blocks___5_____3___bn1_bias, False, 0.1, 0.001);  x_323 = getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = getattr_getattr_l__mod___blocks___5_____3___bn1_weight = getattr_getattr_l__mod___blocks___5_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_325 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_drop(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_327 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_act(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_328 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_329 = torch.nn.functional.batch_norm(x_328, getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn2_running_var, getattr_getattr_l__mod___blocks___5_____3___bn2_weight, getattr_getattr_l__mod___blocks___5_____3___bn2_bias, False, 0.1, 0.001);  x_328 = getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = getattr_getattr_l__mod___blocks___5_____3___bn2_weight = getattr_getattr_l__mod___blocks___5_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_330 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_drop(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_332 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_act(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_333 = self.getattr_getattr_L__mod___blocks___5_____3___se(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_334 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_335 = torch.nn.functional.batch_norm(x_334, getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn3_running_var, getattr_getattr_l__mod___blocks___5_____3___bn3_weight, getattr_getattr_l__mod___blocks___5_____3___bn3_bias, False, 0.1, 0.001);  x_334 = getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = getattr_getattr_l__mod___blocks___5_____3___bn3_weight = getattr_getattr_l__mod___blocks___5_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_336 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_drop(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_338 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_act(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____3___drop_path = self.getattr_getattr_L__mod___blocks___5_____3___drop_path(x_338);  x_338 = None
    shortcut_20 = getattr_getattr_l__mod___blocks___5_____3___drop_path + shortcut_19;  getattr_getattr_l__mod___blocks___5_____3___drop_path = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_340 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pw(shortcut_20);  shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_341 = torch.nn.functional.batch_norm(x_340, getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn1_running_var, getattr_getattr_l__mod___blocks___6_____0___bn1_weight, getattr_getattr_l__mod___blocks___6_____0___bn1_bias, False, 0.1, 0.001);  x_340 = getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = getattr_getattr_l__mod___blocks___6_____0___bn1_weight = getattr_getattr_l__mod___blocks___6_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_342 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_drop(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_344 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_act(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_345 = self.getattr_getattr_L__mod___blocks___6_____0___conv_dw(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_346 = torch.nn.functional.batch_norm(x_345, getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn2_running_var, getattr_getattr_l__mod___blocks___6_____0___bn2_weight, getattr_getattr_l__mod___blocks___6_____0___bn2_bias, False, 0.1, 0.001);  x_345 = getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = getattr_getattr_l__mod___blocks___6_____0___bn2_weight = getattr_getattr_l__mod___blocks___6_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_347 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_drop(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_349 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_act(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_350 = self.getattr_getattr_L__mod___blocks___6_____0___se(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_351 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pwl(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_352 = torch.nn.functional.batch_norm(x_351, getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn3_running_var, getattr_getattr_l__mod___blocks___6_____0___bn3_weight, getattr_getattr_l__mod___blocks___6_____0___bn3_bias, False, 0.1, 0.001);  x_351 = getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = getattr_getattr_l__mod___blocks___6_____0___bn3_weight = getattr_getattr_l__mod___blocks___6_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_353 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_drop(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_356 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_act(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    x_357 = self.L__mod___conv_head(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___bn2_running_mean = self.L__mod___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_var = self.L__mod___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn2_weight = self.L__mod___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn2_bias = self.L__mod___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_358 = torch.nn.functional.batch_norm(x_357, l__mod___bn2_running_mean, l__mod___bn2_running_var, l__mod___bn2_weight, l__mod___bn2_bias, False, 0.1, 0.001);  x_357 = l__mod___bn2_running_mean = l__mod___bn2_running_var = l__mod___bn2_weight = l__mod___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_359 = self.L__mod___bn2_drop(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_362 = self.L__mod___bn2_act(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_363 = self.L__mod___global_pool_pool(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_365 = self.L__mod___global_pool_flatten(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    x_366 = self.L__mod___classifier(x_365);  x_365 = None
    return (x_366,)
    