from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_bn_running_mean = self.L__mod___stem_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_var = self.L__mod___stem_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_bn_weight = self.L__mod___stem_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_bn_bias = self.L__mod___stem_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_bn_running_mean, l__mod___stem_bn_running_var, l__mod___stem_bn_weight, l__mod___stem_bn_bias, False, 0.1, 1e-05);  x = l__mod___stem_bn_running_mean = l__mod___stem_bn_running_var = l__mod___stem_bn_weight = l__mod___stem_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___stem_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_6 = self.L__mod___s1_b1_conv1_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b1_conv1_bn_running_mean = self.L__mod___s1_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv1_bn_running_var = self.L__mod___s1_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv1_bn_weight = self.L__mod___s1_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv1_bn_bias = self.L__mod___s1_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, l__mod___s1_b1_conv1_bn_running_mean, l__mod___s1_b1_conv1_bn_running_var, l__mod___s1_b1_conv1_bn_weight, l__mod___s1_b1_conv1_bn_bias, False, 0.1, 1e-05);  x_6 = l__mod___s1_b1_conv1_bn_running_mean = l__mod___s1_b1_conv1_bn_running_var = l__mod___s1_b1_conv1_bn_weight = l__mod___s1_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.L__mod___s1_b1_conv1_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.L__mod___s1_b1_conv1_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_12 = self.L__mod___s1_b1_conv2_conv(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b1_conv2_bn_running_mean = self.L__mod___s1_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv2_bn_running_var = self.L__mod___s1_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv2_bn_weight = self.L__mod___s1_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv2_bn_bias = self.L__mod___s1_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, l__mod___s1_b1_conv2_bn_running_mean, l__mod___s1_b1_conv2_bn_running_var, l__mod___s1_b1_conv2_bn_weight, l__mod___s1_b1_conv2_bn_bias, False, 0.1, 1e-05);  x_12 = l__mod___s1_b1_conv2_bn_running_mean = l__mod___s1_b1_conv2_bn_running_var = l__mod___s1_b1_conv2_bn_weight = l__mod___s1_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.L__mod___s1_b1_conv2_bn_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.L__mod___s1_b1_conv2_bn_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_17.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_1 = self.L__mod___s1_b1_se_fc1(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s1_b1_se_bn = self.L__mod___s1_b1_se_bn(x_se_1);  x_se_1 = None
    x_se_2 = self.L__mod___s1_b1_se_act(l__mod___s1_b1_se_bn);  l__mod___s1_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_3 = self.L__mod___s1_b1_se_fc2(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid = x_se_3.sigmoid();  x_se_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_18 = x_17 * sigmoid;  x_17 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_19 = self.L__mod___s1_b1_conv3_conv(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b1_conv3_bn_running_mean = self.L__mod___s1_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv3_bn_running_var = self.L__mod___s1_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv3_bn_weight = self.L__mod___s1_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv3_bn_bias = self.L__mod___s1_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, l__mod___s1_b1_conv3_bn_running_mean, l__mod___s1_b1_conv3_bn_running_var, l__mod___s1_b1_conv3_bn_weight, l__mod___s1_b1_conv3_bn_bias, False, 0.1, 1e-05);  x_19 = l__mod___s1_b1_conv3_bn_running_mean = l__mod___s1_b1_conv3_bn_running_var = l__mod___s1_b1_conv3_bn_weight = l__mod___s1_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.L__mod___s1_b1_conv3_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.L__mod___s1_b1_conv3_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s1_b1_drop_path = self.L__mod___s1_b1_drop_path(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_25 = self.L__mod___s1_b1_downsample_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b1_downsample_bn_running_mean = self.L__mod___s1_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_downsample_bn_running_var = self.L__mod___s1_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_downsample_bn_weight = self.L__mod___s1_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_downsample_bn_bias = self.L__mod___s1_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_25, l__mod___s1_b1_downsample_bn_running_mean, l__mod___s1_b1_downsample_bn_running_var, l__mod___s1_b1_downsample_bn_weight, l__mod___s1_b1_downsample_bn_bias, False, 0.1, 1e-05);  x_25 = l__mod___s1_b1_downsample_bn_running_mean = l__mod___s1_b1_downsample_bn_running_var = l__mod___s1_b1_downsample_bn_weight = l__mod___s1_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.L__mod___s1_b1_downsample_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_29 = self.L__mod___s1_b1_downsample_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_30 = l__mod___s1_b1_drop_path + x_29;  l__mod___s1_b1_drop_path = x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_1 = self.L__mod___s1_b1_act3(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_33 = self.L__mod___s1_b2_conv1_conv(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b2_conv1_bn_running_mean = self.L__mod___s1_b2_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b2_conv1_bn_running_var = self.L__mod___s1_b2_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b2_conv1_bn_weight = self.L__mod___s1_b2_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b2_conv1_bn_bias = self.L__mod___s1_b2_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_34 = torch.nn.functional.batch_norm(x_33, l__mod___s1_b2_conv1_bn_running_mean, l__mod___s1_b2_conv1_bn_running_var, l__mod___s1_b2_conv1_bn_weight, l__mod___s1_b2_conv1_bn_bias, False, 0.1, 1e-05);  x_33 = l__mod___s1_b2_conv1_bn_running_mean = l__mod___s1_b2_conv1_bn_running_var = l__mod___s1_b2_conv1_bn_weight = l__mod___s1_b2_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_35 = self.L__mod___s1_b2_conv1_bn_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_38 = self.L__mod___s1_b2_conv1_bn_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_39 = self.L__mod___s1_b2_conv2_conv(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b2_conv2_bn_running_mean = self.L__mod___s1_b2_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b2_conv2_bn_running_var = self.L__mod___s1_b2_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b2_conv2_bn_weight = self.L__mod___s1_b2_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b2_conv2_bn_bias = self.L__mod___s1_b2_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_40 = torch.nn.functional.batch_norm(x_39, l__mod___s1_b2_conv2_bn_running_mean, l__mod___s1_b2_conv2_bn_running_var, l__mod___s1_b2_conv2_bn_weight, l__mod___s1_b2_conv2_bn_bias, False, 0.1, 1e-05);  x_39 = l__mod___s1_b2_conv2_bn_running_mean = l__mod___s1_b2_conv2_bn_running_var = l__mod___s1_b2_conv2_bn_weight = l__mod___s1_b2_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_41 = self.L__mod___s1_b2_conv2_bn_drop(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_44 = self.L__mod___s1_b2_conv2_bn_act(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_44.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_5 = self.L__mod___s1_b2_se_fc1(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s1_b2_se_bn = self.L__mod___s1_b2_se_bn(x_se_5);  x_se_5 = None
    x_se_6 = self.L__mod___s1_b2_se_act(l__mod___s1_b2_se_bn);  l__mod___s1_b2_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_7 = self.L__mod___s1_b2_se_fc2(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = x_se_7.sigmoid();  x_se_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_45 = x_44 * sigmoid_1;  x_44 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_46 = self.L__mod___s1_b2_conv3_conv(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s1_b2_conv3_bn_running_mean = self.L__mod___s1_b2_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b2_conv3_bn_running_var = self.L__mod___s1_b2_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b2_conv3_bn_weight = self.L__mod___s1_b2_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b2_conv3_bn_bias = self.L__mod___s1_b2_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_47 = torch.nn.functional.batch_norm(x_46, l__mod___s1_b2_conv3_bn_running_mean, l__mod___s1_b2_conv3_bn_running_var, l__mod___s1_b2_conv3_bn_weight, l__mod___s1_b2_conv3_bn_bias, False, 0.1, 1e-05);  x_46 = l__mod___s1_b2_conv3_bn_running_mean = l__mod___s1_b2_conv3_bn_running_var = l__mod___s1_b2_conv3_bn_weight = l__mod___s1_b2_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_48 = self.L__mod___s1_b2_conv3_bn_drop(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_51 = self.L__mod___s1_b2_conv3_bn_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s1_b2_drop_path = self.L__mod___s1_b2_drop_path(x_51);  x_51 = None
    l__mod___s1_b2_downsample = self.L__mod___s1_b2_downsample(shortcut_1);  shortcut_1 = None
    x_52 = l__mod___s1_b2_drop_path + l__mod___s1_b2_downsample;  l__mod___s1_b2_drop_path = l__mod___s1_b2_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_2 = self.L__mod___s1_b2_act3(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_56 = self.L__mod___s2_b1_conv1_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b1_conv1_bn_running_mean = self.L__mod___s2_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv1_bn_running_var = self.L__mod___s2_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv1_bn_weight = self.L__mod___s2_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv1_bn_bias = self.L__mod___s2_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_57 = torch.nn.functional.batch_norm(x_56, l__mod___s2_b1_conv1_bn_running_mean, l__mod___s2_b1_conv1_bn_running_var, l__mod___s2_b1_conv1_bn_weight, l__mod___s2_b1_conv1_bn_bias, False, 0.1, 1e-05);  x_56 = l__mod___s2_b1_conv1_bn_running_mean = l__mod___s2_b1_conv1_bn_running_var = l__mod___s2_b1_conv1_bn_weight = l__mod___s2_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_58 = self.L__mod___s2_b1_conv1_bn_drop(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_61 = self.L__mod___s2_b1_conv1_bn_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_62 = self.L__mod___s2_b1_conv2_conv(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b1_conv2_bn_running_mean = self.L__mod___s2_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv2_bn_running_var = self.L__mod___s2_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv2_bn_weight = self.L__mod___s2_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv2_bn_bias = self.L__mod___s2_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_63 = torch.nn.functional.batch_norm(x_62, l__mod___s2_b1_conv2_bn_running_mean, l__mod___s2_b1_conv2_bn_running_var, l__mod___s2_b1_conv2_bn_weight, l__mod___s2_b1_conv2_bn_bias, False, 0.1, 1e-05);  x_62 = l__mod___s2_b1_conv2_bn_running_mean = l__mod___s2_b1_conv2_bn_running_var = l__mod___s2_b1_conv2_bn_weight = l__mod___s2_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_64 = self.L__mod___s2_b1_conv2_bn_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_67 = self.L__mod___s2_b1_conv2_bn_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_67.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_9 = self.L__mod___s2_b1_se_fc1(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b1_se_bn = self.L__mod___s2_b1_se_bn(x_se_9);  x_se_9 = None
    x_se_10 = self.L__mod___s2_b1_se_act(l__mod___s2_b1_se_bn);  l__mod___s2_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_11 = self.L__mod___s2_b1_se_fc2(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = x_se_11.sigmoid();  x_se_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_68 = x_67 * sigmoid_2;  x_67 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_69 = self.L__mod___s2_b1_conv3_conv(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b1_conv3_bn_running_mean = self.L__mod___s2_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv3_bn_running_var = self.L__mod___s2_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv3_bn_weight = self.L__mod___s2_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv3_bn_bias = self.L__mod___s2_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_70 = torch.nn.functional.batch_norm(x_69, l__mod___s2_b1_conv3_bn_running_mean, l__mod___s2_b1_conv3_bn_running_var, l__mod___s2_b1_conv3_bn_weight, l__mod___s2_b1_conv3_bn_bias, False, 0.1, 1e-05);  x_69 = l__mod___s2_b1_conv3_bn_running_mean = l__mod___s2_b1_conv3_bn_running_var = l__mod___s2_b1_conv3_bn_weight = l__mod___s2_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_71 = self.L__mod___s2_b1_conv3_bn_drop(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_74 = self.L__mod___s2_b1_conv3_bn_act(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b1_drop_path = self.L__mod___s2_b1_drop_path(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_75 = self.L__mod___s2_b1_downsample_conv(shortcut_2);  shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b1_downsample_bn_running_mean = self.L__mod___s2_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_downsample_bn_running_var = self.L__mod___s2_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_downsample_bn_weight = self.L__mod___s2_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_downsample_bn_bias = self.L__mod___s2_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_76 = torch.nn.functional.batch_norm(x_75, l__mod___s2_b1_downsample_bn_running_mean, l__mod___s2_b1_downsample_bn_running_var, l__mod___s2_b1_downsample_bn_weight, l__mod___s2_b1_downsample_bn_bias, False, 0.1, 1e-05);  x_75 = l__mod___s2_b1_downsample_bn_running_mean = l__mod___s2_b1_downsample_bn_running_var = l__mod___s2_b1_downsample_bn_weight = l__mod___s2_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_77 = self.L__mod___s2_b1_downsample_bn_drop(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_79 = self.L__mod___s2_b1_downsample_bn_act(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_80 = l__mod___s2_b1_drop_path + x_79;  l__mod___s2_b1_drop_path = x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_3 = self.L__mod___s2_b1_act3(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_83 = self.L__mod___s2_b2_conv1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b2_conv1_bn_running_mean = self.L__mod___s2_b2_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b2_conv1_bn_running_var = self.L__mod___s2_b2_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b2_conv1_bn_weight = self.L__mod___s2_b2_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b2_conv1_bn_bias = self.L__mod___s2_b2_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_84 = torch.nn.functional.batch_norm(x_83, l__mod___s2_b2_conv1_bn_running_mean, l__mod___s2_b2_conv1_bn_running_var, l__mod___s2_b2_conv1_bn_weight, l__mod___s2_b2_conv1_bn_bias, False, 0.1, 1e-05);  x_83 = l__mod___s2_b2_conv1_bn_running_mean = l__mod___s2_b2_conv1_bn_running_var = l__mod___s2_b2_conv1_bn_weight = l__mod___s2_b2_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_85 = self.L__mod___s2_b2_conv1_bn_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_88 = self.L__mod___s2_b2_conv1_bn_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_89 = self.L__mod___s2_b2_conv2_conv(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b2_conv2_bn_running_mean = self.L__mod___s2_b2_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b2_conv2_bn_running_var = self.L__mod___s2_b2_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b2_conv2_bn_weight = self.L__mod___s2_b2_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b2_conv2_bn_bias = self.L__mod___s2_b2_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, l__mod___s2_b2_conv2_bn_running_mean, l__mod___s2_b2_conv2_bn_running_var, l__mod___s2_b2_conv2_bn_weight, l__mod___s2_b2_conv2_bn_bias, False, 0.1, 1e-05);  x_89 = l__mod___s2_b2_conv2_bn_running_mean = l__mod___s2_b2_conv2_bn_running_var = l__mod___s2_b2_conv2_bn_weight = l__mod___s2_b2_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_91 = self.L__mod___s2_b2_conv2_bn_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_94 = self.L__mod___s2_b2_conv2_bn_act(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_94.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_13 = self.L__mod___s2_b2_se_fc1(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b2_se_bn = self.L__mod___s2_b2_se_bn(x_se_13);  x_se_13 = None
    x_se_14 = self.L__mod___s2_b2_se_act(l__mod___s2_b2_se_bn);  l__mod___s2_b2_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_15 = self.L__mod___s2_b2_se_fc2(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = x_se_15.sigmoid();  x_se_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_95 = x_94 * sigmoid_3;  x_94 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_96 = self.L__mod___s2_b2_conv3_conv(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b2_conv3_bn_running_mean = self.L__mod___s2_b2_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b2_conv3_bn_running_var = self.L__mod___s2_b2_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b2_conv3_bn_weight = self.L__mod___s2_b2_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b2_conv3_bn_bias = self.L__mod___s2_b2_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_97 = torch.nn.functional.batch_norm(x_96, l__mod___s2_b2_conv3_bn_running_mean, l__mod___s2_b2_conv3_bn_running_var, l__mod___s2_b2_conv3_bn_weight, l__mod___s2_b2_conv3_bn_bias, False, 0.1, 1e-05);  x_96 = l__mod___s2_b2_conv3_bn_running_mean = l__mod___s2_b2_conv3_bn_running_var = l__mod___s2_b2_conv3_bn_weight = l__mod___s2_b2_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_98 = self.L__mod___s2_b2_conv3_bn_drop(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_101 = self.L__mod___s2_b2_conv3_bn_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b2_drop_path = self.L__mod___s2_b2_drop_path(x_101);  x_101 = None
    l__mod___s2_b2_downsample = self.L__mod___s2_b2_downsample(shortcut_3);  shortcut_3 = None
    x_102 = l__mod___s2_b2_drop_path + l__mod___s2_b2_downsample;  l__mod___s2_b2_drop_path = l__mod___s2_b2_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_4 = self.L__mod___s2_b2_act3(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_105 = self.L__mod___s2_b3_conv1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b3_conv1_bn_running_mean = self.L__mod___s2_b3_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b3_conv1_bn_running_var = self.L__mod___s2_b3_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b3_conv1_bn_weight = self.L__mod___s2_b3_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b3_conv1_bn_bias = self.L__mod___s2_b3_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, l__mod___s2_b3_conv1_bn_running_mean, l__mod___s2_b3_conv1_bn_running_var, l__mod___s2_b3_conv1_bn_weight, l__mod___s2_b3_conv1_bn_bias, False, 0.1, 1e-05);  x_105 = l__mod___s2_b3_conv1_bn_running_mean = l__mod___s2_b3_conv1_bn_running_var = l__mod___s2_b3_conv1_bn_weight = l__mod___s2_b3_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.L__mod___s2_b3_conv1_bn_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_110 = self.L__mod___s2_b3_conv1_bn_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_111 = self.L__mod___s2_b3_conv2_conv(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b3_conv2_bn_running_mean = self.L__mod___s2_b3_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b3_conv2_bn_running_var = self.L__mod___s2_b3_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b3_conv2_bn_weight = self.L__mod___s2_b3_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b3_conv2_bn_bias = self.L__mod___s2_b3_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_112 = torch.nn.functional.batch_norm(x_111, l__mod___s2_b3_conv2_bn_running_mean, l__mod___s2_b3_conv2_bn_running_var, l__mod___s2_b3_conv2_bn_weight, l__mod___s2_b3_conv2_bn_bias, False, 0.1, 1e-05);  x_111 = l__mod___s2_b3_conv2_bn_running_mean = l__mod___s2_b3_conv2_bn_running_var = l__mod___s2_b3_conv2_bn_weight = l__mod___s2_b3_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_113 = self.L__mod___s2_b3_conv2_bn_drop(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_116 = self.L__mod___s2_b3_conv2_bn_act(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_116.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_17 = self.L__mod___s2_b3_se_fc1(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b3_se_bn = self.L__mod___s2_b3_se_bn(x_se_17);  x_se_17 = None
    x_se_18 = self.L__mod___s2_b3_se_act(l__mod___s2_b3_se_bn);  l__mod___s2_b3_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_19 = self.L__mod___s2_b3_se_fc2(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = x_se_19.sigmoid();  x_se_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_117 = x_116 * sigmoid_4;  x_116 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_118 = self.L__mod___s2_b3_conv3_conv(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b3_conv3_bn_running_mean = self.L__mod___s2_b3_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b3_conv3_bn_running_var = self.L__mod___s2_b3_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b3_conv3_bn_weight = self.L__mod___s2_b3_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b3_conv3_bn_bias = self.L__mod___s2_b3_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_118, l__mod___s2_b3_conv3_bn_running_mean, l__mod___s2_b3_conv3_bn_running_var, l__mod___s2_b3_conv3_bn_weight, l__mod___s2_b3_conv3_bn_bias, False, 0.1, 1e-05);  x_118 = l__mod___s2_b3_conv3_bn_running_mean = l__mod___s2_b3_conv3_bn_running_var = l__mod___s2_b3_conv3_bn_weight = l__mod___s2_b3_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.L__mod___s2_b3_conv3_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.L__mod___s2_b3_conv3_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b3_drop_path = self.L__mod___s2_b3_drop_path(x_123);  x_123 = None
    l__mod___s2_b3_downsample = self.L__mod___s2_b3_downsample(shortcut_4);  shortcut_4 = None
    x_124 = l__mod___s2_b3_drop_path + l__mod___s2_b3_downsample;  l__mod___s2_b3_drop_path = l__mod___s2_b3_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_5 = self.L__mod___s2_b3_act3(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_127 = self.L__mod___s2_b4_conv1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b4_conv1_bn_running_mean = self.L__mod___s2_b4_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b4_conv1_bn_running_var = self.L__mod___s2_b4_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b4_conv1_bn_weight = self.L__mod___s2_b4_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b4_conv1_bn_bias = self.L__mod___s2_b4_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_128 = torch.nn.functional.batch_norm(x_127, l__mod___s2_b4_conv1_bn_running_mean, l__mod___s2_b4_conv1_bn_running_var, l__mod___s2_b4_conv1_bn_weight, l__mod___s2_b4_conv1_bn_bias, False, 0.1, 1e-05);  x_127 = l__mod___s2_b4_conv1_bn_running_mean = l__mod___s2_b4_conv1_bn_running_var = l__mod___s2_b4_conv1_bn_weight = l__mod___s2_b4_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_129 = self.L__mod___s2_b4_conv1_bn_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_132 = self.L__mod___s2_b4_conv1_bn_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_133 = self.L__mod___s2_b4_conv2_conv(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b4_conv2_bn_running_mean = self.L__mod___s2_b4_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b4_conv2_bn_running_var = self.L__mod___s2_b4_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b4_conv2_bn_weight = self.L__mod___s2_b4_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b4_conv2_bn_bias = self.L__mod___s2_b4_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, l__mod___s2_b4_conv2_bn_running_mean, l__mod___s2_b4_conv2_bn_running_var, l__mod___s2_b4_conv2_bn_weight, l__mod___s2_b4_conv2_bn_bias, False, 0.1, 1e-05);  x_133 = l__mod___s2_b4_conv2_bn_running_mean = l__mod___s2_b4_conv2_bn_running_var = l__mod___s2_b4_conv2_bn_weight = l__mod___s2_b4_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.L__mod___s2_b4_conv2_bn_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.L__mod___s2_b4_conv2_bn_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_138.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_21 = self.L__mod___s2_b4_se_fc1(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b4_se_bn = self.L__mod___s2_b4_se_bn(x_se_21);  x_se_21 = None
    x_se_22 = self.L__mod___s2_b4_se_act(l__mod___s2_b4_se_bn);  l__mod___s2_b4_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_23 = self.L__mod___s2_b4_se_fc2(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5 = x_se_23.sigmoid();  x_se_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_139 = x_138 * sigmoid_5;  x_138 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_140 = self.L__mod___s2_b4_conv3_conv(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b4_conv3_bn_running_mean = self.L__mod___s2_b4_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b4_conv3_bn_running_var = self.L__mod___s2_b4_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b4_conv3_bn_weight = self.L__mod___s2_b4_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b4_conv3_bn_bias = self.L__mod___s2_b4_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_141 = torch.nn.functional.batch_norm(x_140, l__mod___s2_b4_conv3_bn_running_mean, l__mod___s2_b4_conv3_bn_running_var, l__mod___s2_b4_conv3_bn_weight, l__mod___s2_b4_conv3_bn_bias, False, 0.1, 1e-05);  x_140 = l__mod___s2_b4_conv3_bn_running_mean = l__mod___s2_b4_conv3_bn_running_var = l__mod___s2_b4_conv3_bn_weight = l__mod___s2_b4_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_142 = self.L__mod___s2_b4_conv3_bn_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_145 = self.L__mod___s2_b4_conv3_bn_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b4_drop_path = self.L__mod___s2_b4_drop_path(x_145);  x_145 = None
    l__mod___s2_b4_downsample = self.L__mod___s2_b4_downsample(shortcut_5);  shortcut_5 = None
    x_146 = l__mod___s2_b4_drop_path + l__mod___s2_b4_downsample;  l__mod___s2_b4_drop_path = l__mod___s2_b4_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_6 = self.L__mod___s2_b4_act3(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_149 = self.L__mod___s2_b5_conv1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b5_conv1_bn_running_mean = self.L__mod___s2_b5_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b5_conv1_bn_running_var = self.L__mod___s2_b5_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b5_conv1_bn_weight = self.L__mod___s2_b5_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b5_conv1_bn_bias = self.L__mod___s2_b5_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_150 = torch.nn.functional.batch_norm(x_149, l__mod___s2_b5_conv1_bn_running_mean, l__mod___s2_b5_conv1_bn_running_var, l__mod___s2_b5_conv1_bn_weight, l__mod___s2_b5_conv1_bn_bias, False, 0.1, 1e-05);  x_149 = l__mod___s2_b5_conv1_bn_running_mean = l__mod___s2_b5_conv1_bn_running_var = l__mod___s2_b5_conv1_bn_weight = l__mod___s2_b5_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_151 = self.L__mod___s2_b5_conv1_bn_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.L__mod___s2_b5_conv1_bn_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_155 = self.L__mod___s2_b5_conv2_conv(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b5_conv2_bn_running_mean = self.L__mod___s2_b5_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b5_conv2_bn_running_var = self.L__mod___s2_b5_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b5_conv2_bn_weight = self.L__mod___s2_b5_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b5_conv2_bn_bias = self.L__mod___s2_b5_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_156 = torch.nn.functional.batch_norm(x_155, l__mod___s2_b5_conv2_bn_running_mean, l__mod___s2_b5_conv2_bn_running_var, l__mod___s2_b5_conv2_bn_weight, l__mod___s2_b5_conv2_bn_bias, False, 0.1, 1e-05);  x_155 = l__mod___s2_b5_conv2_bn_running_mean = l__mod___s2_b5_conv2_bn_running_var = l__mod___s2_b5_conv2_bn_weight = l__mod___s2_b5_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_157 = self.L__mod___s2_b5_conv2_bn_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_160 = self.L__mod___s2_b5_conv2_bn_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_160.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_25 = self.L__mod___s2_b5_se_fc1(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b5_se_bn = self.L__mod___s2_b5_se_bn(x_se_25);  x_se_25 = None
    x_se_26 = self.L__mod___s2_b5_se_act(l__mod___s2_b5_se_bn);  l__mod___s2_b5_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_27 = self.L__mod___s2_b5_se_fc2(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6 = x_se_27.sigmoid();  x_se_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_161 = x_160 * sigmoid_6;  x_160 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_162 = self.L__mod___s2_b5_conv3_conv(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s2_b5_conv3_bn_running_mean = self.L__mod___s2_b5_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b5_conv3_bn_running_var = self.L__mod___s2_b5_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b5_conv3_bn_weight = self.L__mod___s2_b5_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b5_conv3_bn_bias = self.L__mod___s2_b5_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_163 = torch.nn.functional.batch_norm(x_162, l__mod___s2_b5_conv3_bn_running_mean, l__mod___s2_b5_conv3_bn_running_var, l__mod___s2_b5_conv3_bn_weight, l__mod___s2_b5_conv3_bn_bias, False, 0.1, 1e-05);  x_162 = l__mod___s2_b5_conv3_bn_running_mean = l__mod___s2_b5_conv3_bn_running_var = l__mod___s2_b5_conv3_bn_weight = l__mod___s2_b5_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_164 = self.L__mod___s2_b5_conv3_bn_drop(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_167 = self.L__mod___s2_b5_conv3_bn_act(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b5_drop_path = self.L__mod___s2_b5_drop_path(x_167);  x_167 = None
    l__mod___s2_b5_downsample = self.L__mod___s2_b5_downsample(shortcut_6);  shortcut_6 = None
    x_168 = l__mod___s2_b5_drop_path + l__mod___s2_b5_downsample;  l__mod___s2_b5_drop_path = l__mod___s2_b5_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_7 = self.L__mod___s2_b5_act3(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_172 = self.L__mod___s3_b1_conv1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b1_conv1_bn_running_mean = self.L__mod___s3_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv1_bn_running_var = self.L__mod___s3_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv1_bn_weight = self.L__mod___s3_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv1_bn_bias = self.L__mod___s3_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_172, l__mod___s3_b1_conv1_bn_running_mean, l__mod___s3_b1_conv1_bn_running_var, l__mod___s3_b1_conv1_bn_weight, l__mod___s3_b1_conv1_bn_bias, False, 0.1, 1e-05);  x_172 = l__mod___s3_b1_conv1_bn_running_mean = l__mod___s3_b1_conv1_bn_running_var = l__mod___s3_b1_conv1_bn_weight = l__mod___s3_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.L__mod___s3_b1_conv1_bn_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_177 = self.L__mod___s3_b1_conv1_bn_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_178 = self.L__mod___s3_b1_conv2_conv(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b1_conv2_bn_running_mean = self.L__mod___s3_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv2_bn_running_var = self.L__mod___s3_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv2_bn_weight = self.L__mod___s3_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv2_bn_bias = self.L__mod___s3_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_179 = torch.nn.functional.batch_norm(x_178, l__mod___s3_b1_conv2_bn_running_mean, l__mod___s3_b1_conv2_bn_running_var, l__mod___s3_b1_conv2_bn_weight, l__mod___s3_b1_conv2_bn_bias, False, 0.1, 1e-05);  x_178 = l__mod___s3_b1_conv2_bn_running_mean = l__mod___s3_b1_conv2_bn_running_var = l__mod___s3_b1_conv2_bn_weight = l__mod___s3_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_180 = self.L__mod___s3_b1_conv2_bn_drop(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_183 = self.L__mod___s3_b1_conv2_bn_act(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_183.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_29 = self.L__mod___s3_b1_se_fc1(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b1_se_bn = self.L__mod___s3_b1_se_bn(x_se_29);  x_se_29 = None
    x_se_30 = self.L__mod___s3_b1_se_act(l__mod___s3_b1_se_bn);  l__mod___s3_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_31 = self.L__mod___s3_b1_se_fc2(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7 = x_se_31.sigmoid();  x_se_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_184 = x_183 * sigmoid_7;  x_183 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_185 = self.L__mod___s3_b1_conv3_conv(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b1_conv3_bn_running_mean = self.L__mod___s3_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv3_bn_running_var = self.L__mod___s3_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv3_bn_weight = self.L__mod___s3_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv3_bn_bias = self.L__mod___s3_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_186 = torch.nn.functional.batch_norm(x_185, l__mod___s3_b1_conv3_bn_running_mean, l__mod___s3_b1_conv3_bn_running_var, l__mod___s3_b1_conv3_bn_weight, l__mod___s3_b1_conv3_bn_bias, False, 0.1, 1e-05);  x_185 = l__mod___s3_b1_conv3_bn_running_mean = l__mod___s3_b1_conv3_bn_running_var = l__mod___s3_b1_conv3_bn_weight = l__mod___s3_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_187 = self.L__mod___s3_b1_conv3_bn_drop(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_190 = self.L__mod___s3_b1_conv3_bn_act(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b1_drop_path = self.L__mod___s3_b1_drop_path(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_191 = self.L__mod___s3_b1_downsample_conv(shortcut_7);  shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b1_downsample_bn_running_mean = self.L__mod___s3_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_downsample_bn_running_var = self.L__mod___s3_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_downsample_bn_weight = self.L__mod___s3_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_downsample_bn_bias = self.L__mod___s3_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_192 = torch.nn.functional.batch_norm(x_191, l__mod___s3_b1_downsample_bn_running_mean, l__mod___s3_b1_downsample_bn_running_var, l__mod___s3_b1_downsample_bn_weight, l__mod___s3_b1_downsample_bn_bias, False, 0.1, 1e-05);  x_191 = l__mod___s3_b1_downsample_bn_running_mean = l__mod___s3_b1_downsample_bn_running_var = l__mod___s3_b1_downsample_bn_weight = l__mod___s3_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_193 = self.L__mod___s3_b1_downsample_bn_drop(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_195 = self.L__mod___s3_b1_downsample_bn_act(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_196 = l__mod___s3_b1_drop_path + x_195;  l__mod___s3_b1_drop_path = x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_8 = self.L__mod___s3_b1_act3(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_199 = self.L__mod___s3_b2_conv1_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b2_conv1_bn_running_mean = self.L__mod___s3_b2_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv1_bn_running_var = self.L__mod___s3_b2_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv1_bn_weight = self.L__mod___s3_b2_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv1_bn_bias = self.L__mod___s3_b2_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_200 = torch.nn.functional.batch_norm(x_199, l__mod___s3_b2_conv1_bn_running_mean, l__mod___s3_b2_conv1_bn_running_var, l__mod___s3_b2_conv1_bn_weight, l__mod___s3_b2_conv1_bn_bias, False, 0.1, 1e-05);  x_199 = l__mod___s3_b2_conv1_bn_running_mean = l__mod___s3_b2_conv1_bn_running_var = l__mod___s3_b2_conv1_bn_weight = l__mod___s3_b2_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_201 = self.L__mod___s3_b2_conv1_bn_drop(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_204 = self.L__mod___s3_b2_conv1_bn_act(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_205 = self.L__mod___s3_b2_conv2_conv(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b2_conv2_bn_running_mean = self.L__mod___s3_b2_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv2_bn_running_var = self.L__mod___s3_b2_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv2_bn_weight = self.L__mod___s3_b2_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv2_bn_bias = self.L__mod___s3_b2_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_205, l__mod___s3_b2_conv2_bn_running_mean, l__mod___s3_b2_conv2_bn_running_var, l__mod___s3_b2_conv2_bn_weight, l__mod___s3_b2_conv2_bn_bias, False, 0.1, 1e-05);  x_205 = l__mod___s3_b2_conv2_bn_running_mean = l__mod___s3_b2_conv2_bn_running_var = l__mod___s3_b2_conv2_bn_weight = l__mod___s3_b2_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.L__mod___s3_b2_conv2_bn_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_210 = self.L__mod___s3_b2_conv2_bn_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_210.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_33 = self.L__mod___s3_b2_se_fc1(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b2_se_bn = self.L__mod___s3_b2_se_bn(x_se_33);  x_se_33 = None
    x_se_34 = self.L__mod___s3_b2_se_act(l__mod___s3_b2_se_bn);  l__mod___s3_b2_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_35 = self.L__mod___s3_b2_se_fc2(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8 = x_se_35.sigmoid();  x_se_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_211 = x_210 * sigmoid_8;  x_210 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_212 = self.L__mod___s3_b2_conv3_conv(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b2_conv3_bn_running_mean = self.L__mod___s3_b2_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv3_bn_running_var = self.L__mod___s3_b2_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv3_bn_weight = self.L__mod___s3_b2_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv3_bn_bias = self.L__mod___s3_b2_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_213 = torch.nn.functional.batch_norm(x_212, l__mod___s3_b2_conv3_bn_running_mean, l__mod___s3_b2_conv3_bn_running_var, l__mod___s3_b2_conv3_bn_weight, l__mod___s3_b2_conv3_bn_bias, False, 0.1, 1e-05);  x_212 = l__mod___s3_b2_conv3_bn_running_mean = l__mod___s3_b2_conv3_bn_running_var = l__mod___s3_b2_conv3_bn_weight = l__mod___s3_b2_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_214 = self.L__mod___s3_b2_conv3_bn_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_217 = self.L__mod___s3_b2_conv3_bn_act(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b2_drop_path = self.L__mod___s3_b2_drop_path(x_217);  x_217 = None
    l__mod___s3_b2_downsample = self.L__mod___s3_b2_downsample(shortcut_8);  shortcut_8 = None
    x_218 = l__mod___s3_b2_drop_path + l__mod___s3_b2_downsample;  l__mod___s3_b2_drop_path = l__mod___s3_b2_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_9 = self.L__mod___s3_b2_act3(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_221 = self.L__mod___s3_b3_conv1_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b3_conv1_bn_running_mean = self.L__mod___s3_b3_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv1_bn_running_var = self.L__mod___s3_b3_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv1_bn_weight = self.L__mod___s3_b3_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv1_bn_bias = self.L__mod___s3_b3_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_222 = torch.nn.functional.batch_norm(x_221, l__mod___s3_b3_conv1_bn_running_mean, l__mod___s3_b3_conv1_bn_running_var, l__mod___s3_b3_conv1_bn_weight, l__mod___s3_b3_conv1_bn_bias, False, 0.1, 1e-05);  x_221 = l__mod___s3_b3_conv1_bn_running_mean = l__mod___s3_b3_conv1_bn_running_var = l__mod___s3_b3_conv1_bn_weight = l__mod___s3_b3_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_223 = self.L__mod___s3_b3_conv1_bn_drop(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_226 = self.L__mod___s3_b3_conv1_bn_act(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.L__mod___s3_b3_conv2_conv(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b3_conv2_bn_running_mean = self.L__mod___s3_b3_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv2_bn_running_var = self.L__mod___s3_b3_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv2_bn_weight = self.L__mod___s3_b3_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv2_bn_bias = self.L__mod___s3_b3_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, l__mod___s3_b3_conv2_bn_running_mean, l__mod___s3_b3_conv2_bn_running_var, l__mod___s3_b3_conv2_bn_weight, l__mod___s3_b3_conv2_bn_bias, False, 0.1, 1e-05);  x_227 = l__mod___s3_b3_conv2_bn_running_mean = l__mod___s3_b3_conv2_bn_running_var = l__mod___s3_b3_conv2_bn_weight = l__mod___s3_b3_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.L__mod___s3_b3_conv2_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_232 = self.L__mod___s3_b3_conv2_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_232.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_37 = self.L__mod___s3_b3_se_fc1(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b3_se_bn = self.L__mod___s3_b3_se_bn(x_se_37);  x_se_37 = None
    x_se_38 = self.L__mod___s3_b3_se_act(l__mod___s3_b3_se_bn);  l__mod___s3_b3_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_39 = self.L__mod___s3_b3_se_fc2(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9 = x_se_39.sigmoid();  x_se_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_233 = x_232 * sigmoid_9;  x_232 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_234 = self.L__mod___s3_b3_conv3_conv(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b3_conv3_bn_running_mean = self.L__mod___s3_b3_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv3_bn_running_var = self.L__mod___s3_b3_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv3_bn_weight = self.L__mod___s3_b3_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv3_bn_bias = self.L__mod___s3_b3_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_235 = torch.nn.functional.batch_norm(x_234, l__mod___s3_b3_conv3_bn_running_mean, l__mod___s3_b3_conv3_bn_running_var, l__mod___s3_b3_conv3_bn_weight, l__mod___s3_b3_conv3_bn_bias, False, 0.1, 1e-05);  x_234 = l__mod___s3_b3_conv3_bn_running_mean = l__mod___s3_b3_conv3_bn_running_var = l__mod___s3_b3_conv3_bn_weight = l__mod___s3_b3_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_236 = self.L__mod___s3_b3_conv3_bn_drop(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_239 = self.L__mod___s3_b3_conv3_bn_act(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b3_drop_path = self.L__mod___s3_b3_drop_path(x_239);  x_239 = None
    l__mod___s3_b3_downsample = self.L__mod___s3_b3_downsample(shortcut_9);  shortcut_9 = None
    x_240 = l__mod___s3_b3_drop_path + l__mod___s3_b3_downsample;  l__mod___s3_b3_drop_path = l__mod___s3_b3_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_10 = self.L__mod___s3_b3_act3(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_243 = self.L__mod___s3_b4_conv1_conv(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b4_conv1_bn_running_mean = self.L__mod___s3_b4_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv1_bn_running_var = self.L__mod___s3_b4_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv1_bn_weight = self.L__mod___s3_b4_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv1_bn_bias = self.L__mod___s3_b4_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_244 = torch.nn.functional.batch_norm(x_243, l__mod___s3_b4_conv1_bn_running_mean, l__mod___s3_b4_conv1_bn_running_var, l__mod___s3_b4_conv1_bn_weight, l__mod___s3_b4_conv1_bn_bias, False, 0.1, 1e-05);  x_243 = l__mod___s3_b4_conv1_bn_running_mean = l__mod___s3_b4_conv1_bn_running_var = l__mod___s3_b4_conv1_bn_weight = l__mod___s3_b4_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_245 = self.L__mod___s3_b4_conv1_bn_drop(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_248 = self.L__mod___s3_b4_conv1_bn_act(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_249 = self.L__mod___s3_b4_conv2_conv(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b4_conv2_bn_running_mean = self.L__mod___s3_b4_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv2_bn_running_var = self.L__mod___s3_b4_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv2_bn_weight = self.L__mod___s3_b4_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv2_bn_bias = self.L__mod___s3_b4_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_250 = torch.nn.functional.batch_norm(x_249, l__mod___s3_b4_conv2_bn_running_mean, l__mod___s3_b4_conv2_bn_running_var, l__mod___s3_b4_conv2_bn_weight, l__mod___s3_b4_conv2_bn_bias, False, 0.1, 1e-05);  x_249 = l__mod___s3_b4_conv2_bn_running_mean = l__mod___s3_b4_conv2_bn_running_var = l__mod___s3_b4_conv2_bn_weight = l__mod___s3_b4_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_251 = self.L__mod___s3_b4_conv2_bn_drop(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_254 = self.L__mod___s3_b4_conv2_bn_act(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_254.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_41 = self.L__mod___s3_b4_se_fc1(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b4_se_bn = self.L__mod___s3_b4_se_bn(x_se_41);  x_se_41 = None
    x_se_42 = self.L__mod___s3_b4_se_act(l__mod___s3_b4_se_bn);  l__mod___s3_b4_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_43 = self.L__mod___s3_b4_se_fc2(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10 = x_se_43.sigmoid();  x_se_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_255 = x_254 * sigmoid_10;  x_254 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_256 = self.L__mod___s3_b4_conv3_conv(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b4_conv3_bn_running_mean = self.L__mod___s3_b4_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv3_bn_running_var = self.L__mod___s3_b4_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv3_bn_weight = self.L__mod___s3_b4_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv3_bn_bias = self.L__mod___s3_b4_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_257 = torch.nn.functional.batch_norm(x_256, l__mod___s3_b4_conv3_bn_running_mean, l__mod___s3_b4_conv3_bn_running_var, l__mod___s3_b4_conv3_bn_weight, l__mod___s3_b4_conv3_bn_bias, False, 0.1, 1e-05);  x_256 = l__mod___s3_b4_conv3_bn_running_mean = l__mod___s3_b4_conv3_bn_running_var = l__mod___s3_b4_conv3_bn_weight = l__mod___s3_b4_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_258 = self.L__mod___s3_b4_conv3_bn_drop(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_261 = self.L__mod___s3_b4_conv3_bn_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b4_drop_path = self.L__mod___s3_b4_drop_path(x_261);  x_261 = None
    l__mod___s3_b4_downsample = self.L__mod___s3_b4_downsample(shortcut_10);  shortcut_10 = None
    x_262 = l__mod___s3_b4_drop_path + l__mod___s3_b4_downsample;  l__mod___s3_b4_drop_path = l__mod___s3_b4_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_11 = self.L__mod___s3_b4_act3(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_265 = self.L__mod___s3_b5_conv1_conv(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b5_conv1_bn_running_mean = self.L__mod___s3_b5_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b5_conv1_bn_running_var = self.L__mod___s3_b5_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b5_conv1_bn_weight = self.L__mod___s3_b5_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b5_conv1_bn_bias = self.L__mod___s3_b5_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_266 = torch.nn.functional.batch_norm(x_265, l__mod___s3_b5_conv1_bn_running_mean, l__mod___s3_b5_conv1_bn_running_var, l__mod___s3_b5_conv1_bn_weight, l__mod___s3_b5_conv1_bn_bias, False, 0.1, 1e-05);  x_265 = l__mod___s3_b5_conv1_bn_running_mean = l__mod___s3_b5_conv1_bn_running_var = l__mod___s3_b5_conv1_bn_weight = l__mod___s3_b5_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_267 = self.L__mod___s3_b5_conv1_bn_drop(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_270 = self.L__mod___s3_b5_conv1_bn_act(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_271 = self.L__mod___s3_b5_conv2_conv(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b5_conv2_bn_running_mean = self.L__mod___s3_b5_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b5_conv2_bn_running_var = self.L__mod___s3_b5_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b5_conv2_bn_weight = self.L__mod___s3_b5_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b5_conv2_bn_bias = self.L__mod___s3_b5_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_272 = torch.nn.functional.batch_norm(x_271, l__mod___s3_b5_conv2_bn_running_mean, l__mod___s3_b5_conv2_bn_running_var, l__mod___s3_b5_conv2_bn_weight, l__mod___s3_b5_conv2_bn_bias, False, 0.1, 1e-05);  x_271 = l__mod___s3_b5_conv2_bn_running_mean = l__mod___s3_b5_conv2_bn_running_var = l__mod___s3_b5_conv2_bn_weight = l__mod___s3_b5_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_273 = self.L__mod___s3_b5_conv2_bn_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_276 = self.L__mod___s3_b5_conv2_bn_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_276.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_45 = self.L__mod___s3_b5_se_fc1(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b5_se_bn = self.L__mod___s3_b5_se_bn(x_se_45);  x_se_45 = None
    x_se_46 = self.L__mod___s3_b5_se_act(l__mod___s3_b5_se_bn);  l__mod___s3_b5_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_47 = self.L__mod___s3_b5_se_fc2(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11 = x_se_47.sigmoid();  x_se_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_277 = x_276 * sigmoid_11;  x_276 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_278 = self.L__mod___s3_b5_conv3_conv(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b5_conv3_bn_running_mean = self.L__mod___s3_b5_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b5_conv3_bn_running_var = self.L__mod___s3_b5_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b5_conv3_bn_weight = self.L__mod___s3_b5_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b5_conv3_bn_bias = self.L__mod___s3_b5_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, l__mod___s3_b5_conv3_bn_running_mean, l__mod___s3_b5_conv3_bn_running_var, l__mod___s3_b5_conv3_bn_weight, l__mod___s3_b5_conv3_bn_bias, False, 0.1, 1e-05);  x_278 = l__mod___s3_b5_conv3_bn_running_mean = l__mod___s3_b5_conv3_bn_running_var = l__mod___s3_b5_conv3_bn_weight = l__mod___s3_b5_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.L__mod___s3_b5_conv3_bn_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_283 = self.L__mod___s3_b5_conv3_bn_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b5_drop_path = self.L__mod___s3_b5_drop_path(x_283);  x_283 = None
    l__mod___s3_b5_downsample = self.L__mod___s3_b5_downsample(shortcut_11);  shortcut_11 = None
    x_284 = l__mod___s3_b5_drop_path + l__mod___s3_b5_downsample;  l__mod___s3_b5_drop_path = l__mod___s3_b5_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_12 = self.L__mod___s3_b5_act3(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_287 = self.L__mod___s3_b6_conv1_conv(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b6_conv1_bn_running_mean = self.L__mod___s3_b6_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b6_conv1_bn_running_var = self.L__mod___s3_b6_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b6_conv1_bn_weight = self.L__mod___s3_b6_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b6_conv1_bn_bias = self.L__mod___s3_b6_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_288 = torch.nn.functional.batch_norm(x_287, l__mod___s3_b6_conv1_bn_running_mean, l__mod___s3_b6_conv1_bn_running_var, l__mod___s3_b6_conv1_bn_weight, l__mod___s3_b6_conv1_bn_bias, False, 0.1, 1e-05);  x_287 = l__mod___s3_b6_conv1_bn_running_mean = l__mod___s3_b6_conv1_bn_running_var = l__mod___s3_b6_conv1_bn_weight = l__mod___s3_b6_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_289 = self.L__mod___s3_b6_conv1_bn_drop(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_292 = self.L__mod___s3_b6_conv1_bn_act(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_293 = self.L__mod___s3_b6_conv2_conv(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b6_conv2_bn_running_mean = self.L__mod___s3_b6_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b6_conv2_bn_running_var = self.L__mod___s3_b6_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b6_conv2_bn_weight = self.L__mod___s3_b6_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b6_conv2_bn_bias = self.L__mod___s3_b6_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_294 = torch.nn.functional.batch_norm(x_293, l__mod___s3_b6_conv2_bn_running_mean, l__mod___s3_b6_conv2_bn_running_var, l__mod___s3_b6_conv2_bn_weight, l__mod___s3_b6_conv2_bn_bias, False, 0.1, 1e-05);  x_293 = l__mod___s3_b6_conv2_bn_running_mean = l__mod___s3_b6_conv2_bn_running_var = l__mod___s3_b6_conv2_bn_weight = l__mod___s3_b6_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_295 = self.L__mod___s3_b6_conv2_bn_drop(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_298 = self.L__mod___s3_b6_conv2_bn_act(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_298.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_49 = self.L__mod___s3_b6_se_fc1(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b6_se_bn = self.L__mod___s3_b6_se_bn(x_se_49);  x_se_49 = None
    x_se_50 = self.L__mod___s3_b6_se_act(l__mod___s3_b6_se_bn);  l__mod___s3_b6_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_51 = self.L__mod___s3_b6_se_fc2(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12 = x_se_51.sigmoid();  x_se_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_299 = x_298 * sigmoid_12;  x_298 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_300 = self.L__mod___s3_b6_conv3_conv(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b6_conv3_bn_running_mean = self.L__mod___s3_b6_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b6_conv3_bn_running_var = self.L__mod___s3_b6_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b6_conv3_bn_weight = self.L__mod___s3_b6_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b6_conv3_bn_bias = self.L__mod___s3_b6_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_301 = torch.nn.functional.batch_norm(x_300, l__mod___s3_b6_conv3_bn_running_mean, l__mod___s3_b6_conv3_bn_running_var, l__mod___s3_b6_conv3_bn_weight, l__mod___s3_b6_conv3_bn_bias, False, 0.1, 1e-05);  x_300 = l__mod___s3_b6_conv3_bn_running_mean = l__mod___s3_b6_conv3_bn_running_var = l__mod___s3_b6_conv3_bn_weight = l__mod___s3_b6_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_302 = self.L__mod___s3_b6_conv3_bn_drop(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_305 = self.L__mod___s3_b6_conv3_bn_act(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b6_drop_path = self.L__mod___s3_b6_drop_path(x_305);  x_305 = None
    l__mod___s3_b6_downsample = self.L__mod___s3_b6_downsample(shortcut_12);  shortcut_12 = None
    x_306 = l__mod___s3_b6_drop_path + l__mod___s3_b6_downsample;  l__mod___s3_b6_drop_path = l__mod___s3_b6_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_13 = self.L__mod___s3_b6_act3(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_309 = self.L__mod___s3_b7_conv1_conv(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b7_conv1_bn_running_mean = self.L__mod___s3_b7_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b7_conv1_bn_running_var = self.L__mod___s3_b7_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b7_conv1_bn_weight = self.L__mod___s3_b7_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b7_conv1_bn_bias = self.L__mod___s3_b7_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_310 = torch.nn.functional.batch_norm(x_309, l__mod___s3_b7_conv1_bn_running_mean, l__mod___s3_b7_conv1_bn_running_var, l__mod___s3_b7_conv1_bn_weight, l__mod___s3_b7_conv1_bn_bias, False, 0.1, 1e-05);  x_309 = l__mod___s3_b7_conv1_bn_running_mean = l__mod___s3_b7_conv1_bn_running_var = l__mod___s3_b7_conv1_bn_weight = l__mod___s3_b7_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_311 = self.L__mod___s3_b7_conv1_bn_drop(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_314 = self.L__mod___s3_b7_conv1_bn_act(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_315 = self.L__mod___s3_b7_conv2_conv(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b7_conv2_bn_running_mean = self.L__mod___s3_b7_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b7_conv2_bn_running_var = self.L__mod___s3_b7_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b7_conv2_bn_weight = self.L__mod___s3_b7_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b7_conv2_bn_bias = self.L__mod___s3_b7_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_316 = torch.nn.functional.batch_norm(x_315, l__mod___s3_b7_conv2_bn_running_mean, l__mod___s3_b7_conv2_bn_running_var, l__mod___s3_b7_conv2_bn_weight, l__mod___s3_b7_conv2_bn_bias, False, 0.1, 1e-05);  x_315 = l__mod___s3_b7_conv2_bn_running_mean = l__mod___s3_b7_conv2_bn_running_var = l__mod___s3_b7_conv2_bn_weight = l__mod___s3_b7_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_317 = self.L__mod___s3_b7_conv2_bn_drop(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_320 = self.L__mod___s3_b7_conv2_bn_act(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_52 = x_320.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_53 = self.L__mod___s3_b7_se_fc1(x_se_52);  x_se_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b7_se_bn = self.L__mod___s3_b7_se_bn(x_se_53);  x_se_53 = None
    x_se_54 = self.L__mod___s3_b7_se_act(l__mod___s3_b7_se_bn);  l__mod___s3_b7_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_55 = self.L__mod___s3_b7_se_fc2(x_se_54);  x_se_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13 = x_se_55.sigmoid();  x_se_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_321 = x_320 * sigmoid_13;  x_320 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_322 = self.L__mod___s3_b7_conv3_conv(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b7_conv3_bn_running_mean = self.L__mod___s3_b7_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b7_conv3_bn_running_var = self.L__mod___s3_b7_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b7_conv3_bn_weight = self.L__mod___s3_b7_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b7_conv3_bn_bias = self.L__mod___s3_b7_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_323 = torch.nn.functional.batch_norm(x_322, l__mod___s3_b7_conv3_bn_running_mean, l__mod___s3_b7_conv3_bn_running_var, l__mod___s3_b7_conv3_bn_weight, l__mod___s3_b7_conv3_bn_bias, False, 0.1, 1e-05);  x_322 = l__mod___s3_b7_conv3_bn_running_mean = l__mod___s3_b7_conv3_bn_running_var = l__mod___s3_b7_conv3_bn_weight = l__mod___s3_b7_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_324 = self.L__mod___s3_b7_conv3_bn_drop(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_327 = self.L__mod___s3_b7_conv3_bn_act(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b7_drop_path = self.L__mod___s3_b7_drop_path(x_327);  x_327 = None
    l__mod___s3_b7_downsample = self.L__mod___s3_b7_downsample(shortcut_13);  shortcut_13 = None
    x_328 = l__mod___s3_b7_drop_path + l__mod___s3_b7_downsample;  l__mod___s3_b7_drop_path = l__mod___s3_b7_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_14 = self.L__mod___s3_b7_act3(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_331 = self.L__mod___s3_b8_conv1_conv(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b8_conv1_bn_running_mean = self.L__mod___s3_b8_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b8_conv1_bn_running_var = self.L__mod___s3_b8_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b8_conv1_bn_weight = self.L__mod___s3_b8_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b8_conv1_bn_bias = self.L__mod___s3_b8_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_332 = torch.nn.functional.batch_norm(x_331, l__mod___s3_b8_conv1_bn_running_mean, l__mod___s3_b8_conv1_bn_running_var, l__mod___s3_b8_conv1_bn_weight, l__mod___s3_b8_conv1_bn_bias, False, 0.1, 1e-05);  x_331 = l__mod___s3_b8_conv1_bn_running_mean = l__mod___s3_b8_conv1_bn_running_var = l__mod___s3_b8_conv1_bn_weight = l__mod___s3_b8_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_333 = self.L__mod___s3_b8_conv1_bn_drop(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_336 = self.L__mod___s3_b8_conv1_bn_act(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_337 = self.L__mod___s3_b8_conv2_conv(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b8_conv2_bn_running_mean = self.L__mod___s3_b8_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b8_conv2_bn_running_var = self.L__mod___s3_b8_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b8_conv2_bn_weight = self.L__mod___s3_b8_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b8_conv2_bn_bias = self.L__mod___s3_b8_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_338 = torch.nn.functional.batch_norm(x_337, l__mod___s3_b8_conv2_bn_running_mean, l__mod___s3_b8_conv2_bn_running_var, l__mod___s3_b8_conv2_bn_weight, l__mod___s3_b8_conv2_bn_bias, False, 0.1, 1e-05);  x_337 = l__mod___s3_b8_conv2_bn_running_mean = l__mod___s3_b8_conv2_bn_running_var = l__mod___s3_b8_conv2_bn_weight = l__mod___s3_b8_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_339 = self.L__mod___s3_b8_conv2_bn_drop(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_342 = self.L__mod___s3_b8_conv2_bn_act(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_56 = x_342.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_57 = self.L__mod___s3_b8_se_fc1(x_se_56);  x_se_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b8_se_bn = self.L__mod___s3_b8_se_bn(x_se_57);  x_se_57 = None
    x_se_58 = self.L__mod___s3_b8_se_act(l__mod___s3_b8_se_bn);  l__mod___s3_b8_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_59 = self.L__mod___s3_b8_se_fc2(x_se_58);  x_se_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_14 = x_se_59.sigmoid();  x_se_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_343 = x_342 * sigmoid_14;  x_342 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_344 = self.L__mod___s3_b8_conv3_conv(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b8_conv3_bn_running_mean = self.L__mod___s3_b8_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b8_conv3_bn_running_var = self.L__mod___s3_b8_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b8_conv3_bn_weight = self.L__mod___s3_b8_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b8_conv3_bn_bias = self.L__mod___s3_b8_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_345 = torch.nn.functional.batch_norm(x_344, l__mod___s3_b8_conv3_bn_running_mean, l__mod___s3_b8_conv3_bn_running_var, l__mod___s3_b8_conv3_bn_weight, l__mod___s3_b8_conv3_bn_bias, False, 0.1, 1e-05);  x_344 = l__mod___s3_b8_conv3_bn_running_mean = l__mod___s3_b8_conv3_bn_running_var = l__mod___s3_b8_conv3_bn_weight = l__mod___s3_b8_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_346 = self.L__mod___s3_b8_conv3_bn_drop(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_349 = self.L__mod___s3_b8_conv3_bn_act(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b8_drop_path = self.L__mod___s3_b8_drop_path(x_349);  x_349 = None
    l__mod___s3_b8_downsample = self.L__mod___s3_b8_downsample(shortcut_14);  shortcut_14 = None
    x_350 = l__mod___s3_b8_drop_path + l__mod___s3_b8_downsample;  l__mod___s3_b8_drop_path = l__mod___s3_b8_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_15 = self.L__mod___s3_b8_act3(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_353 = self.L__mod___s3_b9_conv1_conv(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b9_conv1_bn_running_mean = self.L__mod___s3_b9_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b9_conv1_bn_running_var = self.L__mod___s3_b9_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b9_conv1_bn_weight = self.L__mod___s3_b9_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b9_conv1_bn_bias = self.L__mod___s3_b9_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_354 = torch.nn.functional.batch_norm(x_353, l__mod___s3_b9_conv1_bn_running_mean, l__mod___s3_b9_conv1_bn_running_var, l__mod___s3_b9_conv1_bn_weight, l__mod___s3_b9_conv1_bn_bias, False, 0.1, 1e-05);  x_353 = l__mod___s3_b9_conv1_bn_running_mean = l__mod___s3_b9_conv1_bn_running_var = l__mod___s3_b9_conv1_bn_weight = l__mod___s3_b9_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_355 = self.L__mod___s3_b9_conv1_bn_drop(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_358 = self.L__mod___s3_b9_conv1_bn_act(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_359 = self.L__mod___s3_b9_conv2_conv(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b9_conv2_bn_running_mean = self.L__mod___s3_b9_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b9_conv2_bn_running_var = self.L__mod___s3_b9_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b9_conv2_bn_weight = self.L__mod___s3_b9_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b9_conv2_bn_bias = self.L__mod___s3_b9_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_360 = torch.nn.functional.batch_norm(x_359, l__mod___s3_b9_conv2_bn_running_mean, l__mod___s3_b9_conv2_bn_running_var, l__mod___s3_b9_conv2_bn_weight, l__mod___s3_b9_conv2_bn_bias, False, 0.1, 1e-05);  x_359 = l__mod___s3_b9_conv2_bn_running_mean = l__mod___s3_b9_conv2_bn_running_var = l__mod___s3_b9_conv2_bn_weight = l__mod___s3_b9_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_361 = self.L__mod___s3_b9_conv2_bn_drop(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_364 = self.L__mod___s3_b9_conv2_bn_act(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_60 = x_364.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_61 = self.L__mod___s3_b9_se_fc1(x_se_60);  x_se_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b9_se_bn = self.L__mod___s3_b9_se_bn(x_se_61);  x_se_61 = None
    x_se_62 = self.L__mod___s3_b9_se_act(l__mod___s3_b9_se_bn);  l__mod___s3_b9_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_63 = self.L__mod___s3_b9_se_fc2(x_se_62);  x_se_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_15 = x_se_63.sigmoid();  x_se_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_365 = x_364 * sigmoid_15;  x_364 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_366 = self.L__mod___s3_b9_conv3_conv(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b9_conv3_bn_running_mean = self.L__mod___s3_b9_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b9_conv3_bn_running_var = self.L__mod___s3_b9_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b9_conv3_bn_weight = self.L__mod___s3_b9_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b9_conv3_bn_bias = self.L__mod___s3_b9_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_367 = torch.nn.functional.batch_norm(x_366, l__mod___s3_b9_conv3_bn_running_mean, l__mod___s3_b9_conv3_bn_running_var, l__mod___s3_b9_conv3_bn_weight, l__mod___s3_b9_conv3_bn_bias, False, 0.1, 1e-05);  x_366 = l__mod___s3_b9_conv3_bn_running_mean = l__mod___s3_b9_conv3_bn_running_var = l__mod___s3_b9_conv3_bn_weight = l__mod___s3_b9_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_368 = self.L__mod___s3_b9_conv3_bn_drop(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_371 = self.L__mod___s3_b9_conv3_bn_act(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b9_drop_path = self.L__mod___s3_b9_drop_path(x_371);  x_371 = None
    l__mod___s3_b9_downsample = self.L__mod___s3_b9_downsample(shortcut_15);  shortcut_15 = None
    x_372 = l__mod___s3_b9_drop_path + l__mod___s3_b9_downsample;  l__mod___s3_b9_drop_path = l__mod___s3_b9_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_16 = self.L__mod___s3_b9_act3(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_375 = self.L__mod___s3_b10_conv1_conv(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b10_conv1_bn_running_mean = self.L__mod___s3_b10_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b10_conv1_bn_running_var = self.L__mod___s3_b10_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b10_conv1_bn_weight = self.L__mod___s3_b10_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b10_conv1_bn_bias = self.L__mod___s3_b10_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_376 = torch.nn.functional.batch_norm(x_375, l__mod___s3_b10_conv1_bn_running_mean, l__mod___s3_b10_conv1_bn_running_var, l__mod___s3_b10_conv1_bn_weight, l__mod___s3_b10_conv1_bn_bias, False, 0.1, 1e-05);  x_375 = l__mod___s3_b10_conv1_bn_running_mean = l__mod___s3_b10_conv1_bn_running_var = l__mod___s3_b10_conv1_bn_weight = l__mod___s3_b10_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_377 = self.L__mod___s3_b10_conv1_bn_drop(x_376);  x_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_380 = self.L__mod___s3_b10_conv1_bn_act(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_381 = self.L__mod___s3_b10_conv2_conv(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b10_conv2_bn_running_mean = self.L__mod___s3_b10_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b10_conv2_bn_running_var = self.L__mod___s3_b10_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b10_conv2_bn_weight = self.L__mod___s3_b10_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b10_conv2_bn_bias = self.L__mod___s3_b10_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_382 = torch.nn.functional.batch_norm(x_381, l__mod___s3_b10_conv2_bn_running_mean, l__mod___s3_b10_conv2_bn_running_var, l__mod___s3_b10_conv2_bn_weight, l__mod___s3_b10_conv2_bn_bias, False, 0.1, 1e-05);  x_381 = l__mod___s3_b10_conv2_bn_running_mean = l__mod___s3_b10_conv2_bn_running_var = l__mod___s3_b10_conv2_bn_weight = l__mod___s3_b10_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_383 = self.L__mod___s3_b10_conv2_bn_drop(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_386 = self.L__mod___s3_b10_conv2_bn_act(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_64 = x_386.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_65 = self.L__mod___s3_b10_se_fc1(x_se_64);  x_se_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b10_se_bn = self.L__mod___s3_b10_se_bn(x_se_65);  x_se_65 = None
    x_se_66 = self.L__mod___s3_b10_se_act(l__mod___s3_b10_se_bn);  l__mod___s3_b10_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_67 = self.L__mod___s3_b10_se_fc2(x_se_66);  x_se_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_16 = x_se_67.sigmoid();  x_se_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_387 = x_386 * sigmoid_16;  x_386 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_388 = self.L__mod___s3_b10_conv3_conv(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b10_conv3_bn_running_mean = self.L__mod___s3_b10_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b10_conv3_bn_running_var = self.L__mod___s3_b10_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b10_conv3_bn_weight = self.L__mod___s3_b10_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b10_conv3_bn_bias = self.L__mod___s3_b10_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_389 = torch.nn.functional.batch_norm(x_388, l__mod___s3_b10_conv3_bn_running_mean, l__mod___s3_b10_conv3_bn_running_var, l__mod___s3_b10_conv3_bn_weight, l__mod___s3_b10_conv3_bn_bias, False, 0.1, 1e-05);  x_388 = l__mod___s3_b10_conv3_bn_running_mean = l__mod___s3_b10_conv3_bn_running_var = l__mod___s3_b10_conv3_bn_weight = l__mod___s3_b10_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_390 = self.L__mod___s3_b10_conv3_bn_drop(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_393 = self.L__mod___s3_b10_conv3_bn_act(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b10_drop_path = self.L__mod___s3_b10_drop_path(x_393);  x_393 = None
    l__mod___s3_b10_downsample = self.L__mod___s3_b10_downsample(shortcut_16);  shortcut_16 = None
    x_394 = l__mod___s3_b10_drop_path + l__mod___s3_b10_downsample;  l__mod___s3_b10_drop_path = l__mod___s3_b10_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_17 = self.L__mod___s3_b10_act3(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_397 = self.L__mod___s3_b11_conv1_conv(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b11_conv1_bn_running_mean = self.L__mod___s3_b11_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b11_conv1_bn_running_var = self.L__mod___s3_b11_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b11_conv1_bn_weight = self.L__mod___s3_b11_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b11_conv1_bn_bias = self.L__mod___s3_b11_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_398 = torch.nn.functional.batch_norm(x_397, l__mod___s3_b11_conv1_bn_running_mean, l__mod___s3_b11_conv1_bn_running_var, l__mod___s3_b11_conv1_bn_weight, l__mod___s3_b11_conv1_bn_bias, False, 0.1, 1e-05);  x_397 = l__mod___s3_b11_conv1_bn_running_mean = l__mod___s3_b11_conv1_bn_running_var = l__mod___s3_b11_conv1_bn_weight = l__mod___s3_b11_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_399 = self.L__mod___s3_b11_conv1_bn_drop(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_402 = self.L__mod___s3_b11_conv1_bn_act(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_403 = self.L__mod___s3_b11_conv2_conv(x_402);  x_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b11_conv2_bn_running_mean = self.L__mod___s3_b11_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b11_conv2_bn_running_var = self.L__mod___s3_b11_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b11_conv2_bn_weight = self.L__mod___s3_b11_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b11_conv2_bn_bias = self.L__mod___s3_b11_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_404 = torch.nn.functional.batch_norm(x_403, l__mod___s3_b11_conv2_bn_running_mean, l__mod___s3_b11_conv2_bn_running_var, l__mod___s3_b11_conv2_bn_weight, l__mod___s3_b11_conv2_bn_bias, False, 0.1, 1e-05);  x_403 = l__mod___s3_b11_conv2_bn_running_mean = l__mod___s3_b11_conv2_bn_running_var = l__mod___s3_b11_conv2_bn_weight = l__mod___s3_b11_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_405 = self.L__mod___s3_b11_conv2_bn_drop(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_408 = self.L__mod___s3_b11_conv2_bn_act(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_68 = x_408.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_69 = self.L__mod___s3_b11_se_fc1(x_se_68);  x_se_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b11_se_bn = self.L__mod___s3_b11_se_bn(x_se_69);  x_se_69 = None
    x_se_70 = self.L__mod___s3_b11_se_act(l__mod___s3_b11_se_bn);  l__mod___s3_b11_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_71 = self.L__mod___s3_b11_se_fc2(x_se_70);  x_se_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17 = x_se_71.sigmoid();  x_se_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_409 = x_408 * sigmoid_17;  x_408 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_410 = self.L__mod___s3_b11_conv3_conv(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s3_b11_conv3_bn_running_mean = self.L__mod___s3_b11_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b11_conv3_bn_running_var = self.L__mod___s3_b11_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b11_conv3_bn_weight = self.L__mod___s3_b11_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b11_conv3_bn_bias = self.L__mod___s3_b11_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_411 = torch.nn.functional.batch_norm(x_410, l__mod___s3_b11_conv3_bn_running_mean, l__mod___s3_b11_conv3_bn_running_var, l__mod___s3_b11_conv3_bn_weight, l__mod___s3_b11_conv3_bn_bias, False, 0.1, 1e-05);  x_410 = l__mod___s3_b11_conv3_bn_running_mean = l__mod___s3_b11_conv3_bn_running_var = l__mod___s3_b11_conv3_bn_weight = l__mod___s3_b11_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_412 = self.L__mod___s3_b11_conv3_bn_drop(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_415 = self.L__mod___s3_b11_conv3_bn_act(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b11_drop_path = self.L__mod___s3_b11_drop_path(x_415);  x_415 = None
    l__mod___s3_b11_downsample = self.L__mod___s3_b11_downsample(shortcut_17);  shortcut_17 = None
    x_416 = l__mod___s3_b11_drop_path + l__mod___s3_b11_downsample;  l__mod___s3_b11_drop_path = l__mod___s3_b11_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_18 = self.L__mod___s3_b11_act3(x_416);  x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_420 = self.L__mod___s4_b1_conv1_conv(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s4_b1_conv1_bn_running_mean = self.L__mod___s4_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv1_bn_running_var = self.L__mod___s4_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv1_bn_weight = self.L__mod___s4_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv1_bn_bias = self.L__mod___s4_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_421 = torch.nn.functional.batch_norm(x_420, l__mod___s4_b1_conv1_bn_running_mean, l__mod___s4_b1_conv1_bn_running_var, l__mod___s4_b1_conv1_bn_weight, l__mod___s4_b1_conv1_bn_bias, False, 0.1, 1e-05);  x_420 = l__mod___s4_b1_conv1_bn_running_mean = l__mod___s4_b1_conv1_bn_running_var = l__mod___s4_b1_conv1_bn_weight = l__mod___s4_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_422 = self.L__mod___s4_b1_conv1_bn_drop(x_421);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_425 = self.L__mod___s4_b1_conv1_bn_act(x_422);  x_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_426 = self.L__mod___s4_b1_conv2_conv(x_425);  x_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s4_b1_conv2_bn_running_mean = self.L__mod___s4_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv2_bn_running_var = self.L__mod___s4_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv2_bn_weight = self.L__mod___s4_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv2_bn_bias = self.L__mod___s4_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_427 = torch.nn.functional.batch_norm(x_426, l__mod___s4_b1_conv2_bn_running_mean, l__mod___s4_b1_conv2_bn_running_var, l__mod___s4_b1_conv2_bn_weight, l__mod___s4_b1_conv2_bn_bias, False, 0.1, 1e-05);  x_426 = l__mod___s4_b1_conv2_bn_running_mean = l__mod___s4_b1_conv2_bn_running_var = l__mod___s4_b1_conv2_bn_weight = l__mod___s4_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_428 = self.L__mod___s4_b1_conv2_bn_drop(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_431 = self.L__mod___s4_b1_conv2_bn_act(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_72 = x_431.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_73 = self.L__mod___s4_b1_se_fc1(x_se_72);  x_se_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b1_se_bn = self.L__mod___s4_b1_se_bn(x_se_73);  x_se_73 = None
    x_se_74 = self.L__mod___s4_b1_se_act(l__mod___s4_b1_se_bn);  l__mod___s4_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_75 = self.L__mod___s4_b1_se_fc2(x_se_74);  x_se_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18 = x_se_75.sigmoid();  x_se_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_432 = x_431 * sigmoid_18;  x_431 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_433 = self.L__mod___s4_b1_conv3_conv(x_432);  x_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s4_b1_conv3_bn_running_mean = self.L__mod___s4_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv3_bn_running_var = self.L__mod___s4_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv3_bn_weight = self.L__mod___s4_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv3_bn_bias = self.L__mod___s4_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_434 = torch.nn.functional.batch_norm(x_433, l__mod___s4_b1_conv3_bn_running_mean, l__mod___s4_b1_conv3_bn_running_var, l__mod___s4_b1_conv3_bn_weight, l__mod___s4_b1_conv3_bn_bias, False, 0.1, 1e-05);  x_433 = l__mod___s4_b1_conv3_bn_running_mean = l__mod___s4_b1_conv3_bn_running_var = l__mod___s4_b1_conv3_bn_weight = l__mod___s4_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_435 = self.L__mod___s4_b1_conv3_bn_drop(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_438 = self.L__mod___s4_b1_conv3_bn_act(x_435);  x_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b1_drop_path = self.L__mod___s4_b1_drop_path(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_439 = self.L__mod___s4_b1_downsample_conv(shortcut_18);  shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___s4_b1_downsample_bn_running_mean = self.L__mod___s4_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_downsample_bn_running_var = self.L__mod___s4_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_downsample_bn_weight = self.L__mod___s4_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_downsample_bn_bias = self.L__mod___s4_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_440 = torch.nn.functional.batch_norm(x_439, l__mod___s4_b1_downsample_bn_running_mean, l__mod___s4_b1_downsample_bn_running_var, l__mod___s4_b1_downsample_bn_weight, l__mod___s4_b1_downsample_bn_bias, False, 0.1, 1e-05);  x_439 = l__mod___s4_b1_downsample_bn_running_mean = l__mod___s4_b1_downsample_bn_running_var = l__mod___s4_b1_downsample_bn_weight = l__mod___s4_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_441 = self.L__mod___s4_b1_downsample_bn_drop(x_440);  x_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_443 = self.L__mod___s4_b1_downsample_bn_act(x_441);  x_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_444 = l__mod___s4_b1_drop_path + x_443;  l__mod___s4_b1_drop_path = x_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    x_447 = self.L__mod___s4_b1_act3(x_444);  x_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:524, code: x = self.final_conv(x)
    x_449 = self.L__mod___final_conv(x_447);  x_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_450 = self.L__mod___head_global_pool_pool(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_452 = self.L__mod___head_global_pool_flatten(x_450);  x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_453 = self.L__mod___head_drop(x_452);  x_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_454 = self.L__mod___head_fc(x_453);  x_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_455 = self.L__mod___head_flatten(x_454);  x_454 = None
    return (x_455,)
    