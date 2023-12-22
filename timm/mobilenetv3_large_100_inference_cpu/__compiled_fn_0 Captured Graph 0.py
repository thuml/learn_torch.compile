from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
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
    x_5 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(shortcut)
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:125, code: x = self.se(x)
    x_10 = self.getattr_getattr_L__mod___blocks___0_____0___se(x_9);  x_9 = None
    
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
    x_15 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___0_____0___drop_path = self.getattr_getattr_L__mod___blocks___0_____0___drop_path(x_15);  x_15 = None
    shortcut_1 = getattr_getattr_l__mod___blocks___0_____0___drop_path + shortcut;  getattr_getattr_l__mod___blocks___0_____0___drop_path = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_17 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_18 = torch.nn.functional.batch_norm(x_17, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, False, 0.1, 1e-05);  x_17 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_19 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_21 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_22 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_23 = torch.nn.functional.batch_norm(x_22, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, False, 0.1, 1e-05);  x_22 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_24 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_26 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_27 = self.getattr_getattr_L__mod___blocks___1_____0___se(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_28 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_29 = torch.nn.functional.batch_norm(x_28, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, False, 0.1, 1e-05);  x_28 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_30 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_33 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_34 = torch.nn.functional.batch_norm(x_33, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, False, 0.1, 1e-05);  x_33 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_35 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_37 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_38 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, False, 0.1, 1e-05);  x_38 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_42 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_43 = self.getattr_getattr_L__mod___blocks___1_____1___se(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_44 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, False, 0.1, 1e-05);  x_44 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_48 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_48);  x_48 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_50 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_3);  shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_51 = torch.nn.functional.batch_norm(x_50, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, False, 0.1, 1e-05);  x_50 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_52 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_54 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_55 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_56 = torch.nn.functional.batch_norm(x_55, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, False, 0.1, 1e-05);  x_55 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_57 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_59 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_59.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___2_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____0___se_gate = self.getattr_getattr_L__mod___blocks___2_____0___se_gate(x_se_3);  x_se_3 = None
    x_60 = x_59 * getattr_getattr_l__mod___blocks___2_____0___se_gate;  x_59 = getattr_getattr_l__mod___blocks___2_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_61 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_62 = torch.nn.functional.batch_norm(x_61, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, False, 0.1, 1e-05);  x_61 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_63 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_66 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_67 = torch.nn.functional.batch_norm(x_66, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, False, 0.1, 1e-05);  x_66 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_68 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_70 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_71 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_72 = torch.nn.functional.batch_norm(x_71, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, False, 0.1, 1e-05);  x_71 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_73 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_75 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_75.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___2_____1___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____1___se_gate = self.getattr_getattr_L__mod___blocks___2_____1___se_gate(x_se_7);  x_se_7 = None
    x_76 = x_75 * getattr_getattr_l__mod___blocks___2_____1___se_gate;  x_75 = getattr_getattr_l__mod___blocks___2_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_77 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_78 = torch.nn.functional.batch_norm(x_77, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, False, 0.1, 1e-05);  x_77 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_79 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_81 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_81);  x_81 = None
    shortcut_5 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_4;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_83 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_84 = torch.nn.functional.batch_norm(x_83, getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn1_running_var, getattr_getattr_l__mod___blocks___2_____2___bn1_weight, getattr_getattr_l__mod___blocks___2_____2___bn1_bias, False, 0.1, 1e-05);  x_83 = getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = getattr_getattr_l__mod___blocks___2_____2___bn1_weight = getattr_getattr_l__mod___blocks___2_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_85 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_87 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_88 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_88, getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn2_running_var, getattr_getattr_l__mod___blocks___2_____2___bn2_weight, getattr_getattr_l__mod___blocks___2_____2___bn2_bias, False, 0.1, 1e-05);  x_88 = getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = getattr_getattr_l__mod___blocks___2_____2___bn2_weight = getattr_getattr_l__mod___blocks___2_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_92 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_92.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___2_____2___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____2___se_gate = self.getattr_getattr_L__mod___blocks___2_____2___se_gate(x_se_11);  x_se_11 = None
    x_93 = x_92 * getattr_getattr_l__mod___blocks___2_____2___se_gate;  x_92 = getattr_getattr_l__mod___blocks___2_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_94 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_95 = torch.nn.functional.batch_norm(x_94, getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn3_running_var, getattr_getattr_l__mod___blocks___2_____2___bn3_weight, getattr_getattr_l__mod___blocks___2_____2___bn3_bias, False, 0.1, 1e-05);  x_94 = getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = getattr_getattr_l__mod___blocks___2_____2___bn3_weight = getattr_getattr_l__mod___blocks___2_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_96 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_98 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____2___drop_path = self.getattr_getattr_L__mod___blocks___2_____2___drop_path(x_98);  x_98 = None
    shortcut_6 = getattr_getattr_l__mod___blocks___2_____2___drop_path + shortcut_5;  getattr_getattr_l__mod___blocks___2_____2___drop_path = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_100 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_6);  shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_101 = torch.nn.functional.batch_norm(x_100, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, False, 0.1, 1e-05);  x_100 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_102 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_104 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_105 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, False, 0.1, 1e-05);  x_105 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_109 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_110 = self.getattr_getattr_L__mod___blocks___3_____0___se(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_111 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_112 = torch.nn.functional.batch_norm(x_111, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, False, 0.1, 1e-05);  x_111 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_113 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_7 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_116 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_117 = torch.nn.functional.batch_norm(x_116, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, False, 0.1, 1e-05);  x_116 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_118 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_120 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_121 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_122 = torch.nn.functional.batch_norm(x_121, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, False, 0.1, 1e-05);  x_121 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_123 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_125 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_126 = self.getattr_getattr_L__mod___blocks___3_____1___se(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_127 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_128 = torch.nn.functional.batch_norm(x_127, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, False, 0.1, 1e-05);  x_127 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_129 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_131 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_131);  x_131 = None
    shortcut_8 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_7;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_133 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, False, 0.1, 1e-05);  x_133 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_137 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_138 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_139 = torch.nn.functional.batch_norm(x_138, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, False, 0.1, 1e-05);  x_138 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_140 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_142 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_143 = self.getattr_getattr_L__mod___blocks___3_____2___se(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_144 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_145 = torch.nn.functional.batch_norm(x_144, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, False, 0.1, 1e-05);  x_144 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_146 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_148 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_148);  x_148 = None
    shortcut_9 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_8;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_150 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_151 = torch.nn.functional.batch_norm(x_150, getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn1_running_var, getattr_getattr_l__mod___blocks___3_____3___bn1_weight, getattr_getattr_l__mod___blocks___3_____3___bn1_bias, False, 0.1, 1e-05);  x_150 = getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = getattr_getattr_l__mod___blocks___3_____3___bn1_weight = getattr_getattr_l__mod___blocks___3_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_155 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_156 = torch.nn.functional.batch_norm(x_155, getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn2_running_var, getattr_getattr_l__mod___blocks___3_____3___bn2_weight, getattr_getattr_l__mod___blocks___3_____3___bn2_bias, False, 0.1, 1e-05);  x_155 = getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = getattr_getattr_l__mod___blocks___3_____3___bn2_weight = getattr_getattr_l__mod___blocks___3_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_157 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_159 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_160 = self.getattr_getattr_L__mod___blocks___3_____3___se(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_161 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_162 = torch.nn.functional.batch_norm(x_161, getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn3_running_var, getattr_getattr_l__mod___blocks___3_____3___bn3_weight, getattr_getattr_l__mod___blocks___3_____3___bn3_bias, False, 0.1, 1e-05);  x_161 = getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = getattr_getattr_l__mod___blocks___3_____3___bn3_weight = getattr_getattr_l__mod___blocks___3_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_163 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_drop(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_165 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____3___drop_path = self.getattr_getattr_L__mod___blocks___3_____3___drop_path(x_165);  x_165 = None
    shortcut_10 = getattr_getattr_l__mod___blocks___3_____3___drop_path + shortcut_9;  getattr_getattr_l__mod___blocks___3_____3___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_167 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_10);  shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_168 = torch.nn.functional.batch_norm(x_167, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, False, 0.1, 1e-05);  x_167 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_169 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_171 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_172 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_172, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, False, 0.1, 1e-05);  x_172 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_176 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_176.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_15);  x_se_15 = None
    x_177 = x_176 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_176 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_178 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_179 = torch.nn.functional.batch_norm(x_178, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, False, 0.1, 1e-05);  x_178 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_180 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_11 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_183 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_184 = torch.nn.functional.batch_norm(x_183, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, False, 0.1, 1e-05);  x_183 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_185 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_188 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_189 = torch.nn.functional.batch_norm(x_188, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, False, 0.1, 1e-05);  x_188 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_190 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_192 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_192.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___4_____1___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____1___se_gate = self.getattr_getattr_L__mod___blocks___4_____1___se_gate(x_se_19);  x_se_19 = None
    x_193 = x_192 * getattr_getattr_l__mod___blocks___4_____1___se_gate;  x_192 = getattr_getattr_l__mod___blocks___4_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_194 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_195 = torch.nn.functional.batch_norm(x_194, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, False, 0.1, 1e-05);  x_194 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_196 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_198 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_198);  x_198 = None
    shortcut_12 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_11;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_200 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_12);  shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_201 = torch.nn.functional.batch_norm(x_200, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, False, 0.1, 1e-05);  x_200 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_202 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_204 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_205 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_205, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, False, 0.1, 1e-05);  x_205 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_209 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_209.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___5_____0___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____0___se_gate = self.getattr_getattr_L__mod___blocks___5_____0___se_gate(x_se_23);  x_se_23 = None
    x_210 = x_209 * getattr_getattr_l__mod___blocks___5_____0___se_gate;  x_209 = getattr_getattr_l__mod___blocks___5_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_211 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_212 = torch.nn.functional.batch_norm(x_211, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, False, 0.1, 1e-05);  x_211 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_213 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_13 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_216 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_217 = torch.nn.functional.batch_norm(x_216, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, False, 0.1, 1e-05);  x_216 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_218 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_220 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_221 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_222 = torch.nn.functional.batch_norm(x_221, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, False, 0.1, 1e-05);  x_221 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_223 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_225 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_225.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___5_____1___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____1___se_gate = self.getattr_getattr_L__mod___blocks___5_____1___se_gate(x_se_27);  x_se_27 = None
    x_226 = x_225 * getattr_getattr_l__mod___blocks___5_____1___se_gate;  x_225 = getattr_getattr_l__mod___blocks___5_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_227 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, False, 0.1, 1e-05);  x_227 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_231 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_231);  x_231 = None
    shortcut_14 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_13;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_233 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_234 = torch.nn.functional.batch_norm(x_233, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, False, 0.1, 1e-05);  x_233 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_235 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_237 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_238 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_239 = torch.nn.functional.batch_norm(x_238, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, False, 0.1, 1e-05);  x_238 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_240 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_242 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_242.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_29 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_reduce(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_30 = self.getattr_getattr_L__mod___blocks___5_____2___se_act1(x_se_29);  x_se_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_31 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_expand(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____2___se_gate = self.getattr_getattr_L__mod___blocks___5_____2___se_gate(x_se_31);  x_se_31 = None
    x_243 = x_242 * getattr_getattr_l__mod___blocks___5_____2___se_gate;  x_242 = getattr_getattr_l__mod___blocks___5_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_244 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_245 = torch.nn.functional.batch_norm(x_244, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, False, 0.1, 1e-05);  x_244 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_246 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_248 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_248);  x_248 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    x_250 = self.getattr_getattr_L__mod___blocks___6_____0___conv(shortcut_15);  shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_251 = torch.nn.functional.batch_norm(x_250, getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn1_running_var, getattr_getattr_l__mod___blocks___6_____0___bn1_weight, getattr_getattr_l__mod___blocks___6_____0___bn1_bias, False, 0.1, 1e-05);  x_250 = getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = getattr_getattr_l__mod___blocks___6_____0___bn1_weight = getattr_getattr_l__mod___blocks___6_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_252 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_drop(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_256 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_257 = self.L__mod___global_pool_pool(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_259 = self.L__mod___global_pool_flatten(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    x_260 = self.L__mod___conv_head(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    x_261 = self.L__mod___act2(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    x_262 = self.L__mod___flatten(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    l__mod___classifier_weight = self.L__mod___classifier_weight
    l__mod___classifier_bias = self.L__mod___classifier_bias
    x_263 = torch._C._nn.linear(x_262, l__mod___classifier_weight, l__mod___classifier_bias);  x_262 = l__mod___classifier_weight = l__mod___classifier_bias = None
    return (x_263,)
    