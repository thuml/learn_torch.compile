from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___stem_bn_num_batches_tracked = self.L__mod___stem_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___stem_bn_num_batches_tracked.add_(1);  l__mod___stem_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_mean = self.L__mod___stem_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_var = self.L__mod___stem_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_bn_weight = self.L__mod___stem_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_bn_bias = self.L__mod___stem_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_bn_running_mean, l__mod___stem_bn_running_var, l__mod___stem_bn_weight, l__mod___stem_bn_bias, True, 0.1, 1e-05);  x = l__mod___stem_bn_running_mean = l__mod___stem_bn_running_var = l__mod___stem_bn_weight = l__mod___stem_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___stem_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_6 = self.L__mod___s1_b1_conv1_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s1_b1_conv1_bn_num_batches_tracked = self.L__mod___s1_b1_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = l__mod___s1_b1_conv1_bn_num_batches_tracked.add_(1);  l__mod___s1_b1_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv1_bn_running_mean = self.L__mod___s1_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv1_bn_running_var = self.L__mod___s1_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv1_bn_weight = self.L__mod___s1_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv1_bn_bias = self.L__mod___s1_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, l__mod___s1_b1_conv1_bn_running_mean, l__mod___s1_b1_conv1_bn_running_var, l__mod___s1_b1_conv1_bn_weight, l__mod___s1_b1_conv1_bn_bias, True, 0.1, 1e-05);  x_6 = l__mod___s1_b1_conv1_bn_running_mean = l__mod___s1_b1_conv1_bn_running_var = l__mod___s1_b1_conv1_bn_weight = l__mod___s1_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.L__mod___s1_b1_conv1_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.L__mod___s1_b1_conv1_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_12 = self.L__mod___s1_b1_conv2_conv(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s1_b1_conv2_bn_num_batches_tracked = self.L__mod___s1_b1_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = l__mod___s1_b1_conv2_bn_num_batches_tracked.add_(1);  l__mod___s1_b1_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv2_bn_running_mean = self.L__mod___s1_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv2_bn_running_var = self.L__mod___s1_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv2_bn_weight = self.L__mod___s1_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv2_bn_bias = self.L__mod___s1_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, l__mod___s1_b1_conv2_bn_running_mean, l__mod___s1_b1_conv2_bn_running_var, l__mod___s1_b1_conv2_bn_weight, l__mod___s1_b1_conv2_bn_bias, True, 0.1, 1e-05);  x_12 = l__mod___s1_b1_conv2_bn_running_mean = l__mod___s1_b1_conv2_bn_running_var = l__mod___s1_b1_conv2_bn_weight = l__mod___s1_b1_conv2_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s1_b1_conv3_bn_num_batches_tracked = self.L__mod___s1_b1_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = l__mod___s1_b1_conv3_bn_num_batches_tracked.add_(1);  l__mod___s1_b1_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv3_bn_running_mean = self.L__mod___s1_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_conv3_bn_running_var = self.L__mod___s1_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_conv3_bn_weight = self.L__mod___s1_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_conv3_bn_bias = self.L__mod___s1_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, l__mod___s1_b1_conv3_bn_running_mean, l__mod___s1_b1_conv3_bn_running_var, l__mod___s1_b1_conv3_bn_weight, l__mod___s1_b1_conv3_bn_bias, True, 0.1, 1e-05);  x_19 = l__mod___s1_b1_conv3_bn_running_mean = l__mod___s1_b1_conv3_bn_running_var = l__mod___s1_b1_conv3_bn_weight = l__mod___s1_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.L__mod___s1_b1_conv3_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.L__mod___s1_b1_conv3_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s1_b1_drop_path = self.L__mod___s1_b1_drop_path(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_25 = self.L__mod___s1_b1_downsample_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s1_b1_downsample_bn_num_batches_tracked = self.L__mod___s1_b1_downsample_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = l__mod___s1_b1_downsample_bn_num_batches_tracked.add_(1);  l__mod___s1_b1_downsample_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_downsample_bn_running_mean = self.L__mod___s1_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s1_b1_downsample_bn_running_var = self.L__mod___s1_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s1_b1_downsample_bn_weight = self.L__mod___s1_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s1_b1_downsample_bn_bias = self.L__mod___s1_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_25, l__mod___s1_b1_downsample_bn_running_mean, l__mod___s1_b1_downsample_bn_running_var, l__mod___s1_b1_downsample_bn_weight, l__mod___s1_b1_downsample_bn_bias, True, 0.1, 1e-05);  x_25 = l__mod___s1_b1_downsample_bn_running_mean = l__mod___s1_b1_downsample_bn_running_var = l__mod___s1_b1_downsample_bn_weight = l__mod___s1_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.L__mod___s1_b1_downsample_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_29 = self.L__mod___s1_b1_downsample_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_30 = l__mod___s1_b1_drop_path + x_29;  l__mod___s1_b1_drop_path = x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_1 = self.L__mod___s1_b1_act3(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_34 = self.L__mod___s2_b1_conv1_conv(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s2_b1_conv1_bn_num_batches_tracked = self.L__mod___s2_b1_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = l__mod___s2_b1_conv1_bn_num_batches_tracked.add_(1);  l__mod___s2_b1_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv1_bn_running_mean = self.L__mod___s2_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv1_bn_running_var = self.L__mod___s2_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv1_bn_weight = self.L__mod___s2_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv1_bn_bias = self.L__mod___s2_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_35 = torch.nn.functional.batch_norm(x_34, l__mod___s2_b1_conv1_bn_running_mean, l__mod___s2_b1_conv1_bn_running_var, l__mod___s2_b1_conv1_bn_weight, l__mod___s2_b1_conv1_bn_bias, True, 0.1, 1e-05);  x_34 = l__mod___s2_b1_conv1_bn_running_mean = l__mod___s2_b1_conv1_bn_running_var = l__mod___s2_b1_conv1_bn_weight = l__mod___s2_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_36 = self.L__mod___s2_b1_conv1_bn_drop(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_39 = self.L__mod___s2_b1_conv1_bn_act(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_40 = self.L__mod___s2_b1_conv2_conv(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s2_b1_conv2_bn_num_batches_tracked = self.L__mod___s2_b1_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = l__mod___s2_b1_conv2_bn_num_batches_tracked.add_(1);  l__mod___s2_b1_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv2_bn_running_mean = self.L__mod___s2_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv2_bn_running_var = self.L__mod___s2_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv2_bn_weight = self.L__mod___s2_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv2_bn_bias = self.L__mod___s2_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_41 = torch.nn.functional.batch_norm(x_40, l__mod___s2_b1_conv2_bn_running_mean, l__mod___s2_b1_conv2_bn_running_var, l__mod___s2_b1_conv2_bn_weight, l__mod___s2_b1_conv2_bn_bias, True, 0.1, 1e-05);  x_40 = l__mod___s2_b1_conv2_bn_running_mean = l__mod___s2_b1_conv2_bn_running_var = l__mod___s2_b1_conv2_bn_weight = l__mod___s2_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_42 = self.L__mod___s2_b1_conv2_bn_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_45 = self.L__mod___s2_b1_conv2_bn_act(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_45.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_5 = self.L__mod___s2_b1_se_fc1(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s2_b1_se_bn = self.L__mod___s2_b1_se_bn(x_se_5);  x_se_5 = None
    x_se_6 = self.L__mod___s2_b1_se_act(l__mod___s2_b1_se_bn);  l__mod___s2_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_7 = self.L__mod___s2_b1_se_fc2(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = x_se_7.sigmoid();  x_se_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_46 = x_45 * sigmoid_1;  x_45 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_47 = self.L__mod___s2_b1_conv3_conv(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s2_b1_conv3_bn_num_batches_tracked = self.L__mod___s2_b1_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = l__mod___s2_b1_conv3_bn_num_batches_tracked.add_(1);  l__mod___s2_b1_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv3_bn_running_mean = self.L__mod___s2_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_conv3_bn_running_var = self.L__mod___s2_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_conv3_bn_weight = self.L__mod___s2_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_conv3_bn_bias = self.L__mod___s2_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_48 = torch.nn.functional.batch_norm(x_47, l__mod___s2_b1_conv3_bn_running_mean, l__mod___s2_b1_conv3_bn_running_var, l__mod___s2_b1_conv3_bn_weight, l__mod___s2_b1_conv3_bn_bias, True, 0.1, 1e-05);  x_47 = l__mod___s2_b1_conv3_bn_running_mean = l__mod___s2_b1_conv3_bn_running_var = l__mod___s2_b1_conv3_bn_weight = l__mod___s2_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_49 = self.L__mod___s2_b1_conv3_bn_drop(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_52 = self.L__mod___s2_b1_conv3_bn_act(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s2_b1_drop_path = self.L__mod___s2_b1_drop_path(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_53 = self.L__mod___s2_b1_downsample_conv(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s2_b1_downsample_bn_num_batches_tracked = self.L__mod___s2_b1_downsample_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = l__mod___s2_b1_downsample_bn_num_batches_tracked.add_(1);  l__mod___s2_b1_downsample_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_downsample_bn_running_mean = self.L__mod___s2_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s2_b1_downsample_bn_running_var = self.L__mod___s2_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s2_b1_downsample_bn_weight = self.L__mod___s2_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s2_b1_downsample_bn_bias = self.L__mod___s2_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_54 = torch.nn.functional.batch_norm(x_53, l__mod___s2_b1_downsample_bn_running_mean, l__mod___s2_b1_downsample_bn_running_var, l__mod___s2_b1_downsample_bn_weight, l__mod___s2_b1_downsample_bn_bias, True, 0.1, 1e-05);  x_53 = l__mod___s2_b1_downsample_bn_running_mean = l__mod___s2_b1_downsample_bn_running_var = l__mod___s2_b1_downsample_bn_weight = l__mod___s2_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_55 = self.L__mod___s2_b1_downsample_bn_drop(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_57 = self.L__mod___s2_b1_downsample_bn_act(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_58 = l__mod___s2_b1_drop_path + x_57;  l__mod___s2_b1_drop_path = x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_2 = self.L__mod___s2_b1_act3(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_62 = self.L__mod___s3_b1_conv1_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b1_conv1_bn_num_batches_tracked = self.L__mod___s3_b1_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = l__mod___s3_b1_conv1_bn_num_batches_tracked.add_(1);  l__mod___s3_b1_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv1_bn_running_mean = self.L__mod___s3_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv1_bn_running_var = self.L__mod___s3_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv1_bn_weight = self.L__mod___s3_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv1_bn_bias = self.L__mod___s3_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_63 = torch.nn.functional.batch_norm(x_62, l__mod___s3_b1_conv1_bn_running_mean, l__mod___s3_b1_conv1_bn_running_var, l__mod___s3_b1_conv1_bn_weight, l__mod___s3_b1_conv1_bn_bias, True, 0.1, 1e-05);  x_62 = l__mod___s3_b1_conv1_bn_running_mean = l__mod___s3_b1_conv1_bn_running_var = l__mod___s3_b1_conv1_bn_weight = l__mod___s3_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_64 = self.L__mod___s3_b1_conv1_bn_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_67 = self.L__mod___s3_b1_conv1_bn_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_68 = self.L__mod___s3_b1_conv2_conv(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b1_conv2_bn_num_batches_tracked = self.L__mod___s3_b1_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = l__mod___s3_b1_conv2_bn_num_batches_tracked.add_(1);  l__mod___s3_b1_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv2_bn_running_mean = self.L__mod___s3_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv2_bn_running_var = self.L__mod___s3_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv2_bn_weight = self.L__mod___s3_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv2_bn_bias = self.L__mod___s3_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_69 = torch.nn.functional.batch_norm(x_68, l__mod___s3_b1_conv2_bn_running_mean, l__mod___s3_b1_conv2_bn_running_var, l__mod___s3_b1_conv2_bn_weight, l__mod___s3_b1_conv2_bn_bias, True, 0.1, 1e-05);  x_68 = l__mod___s3_b1_conv2_bn_running_mean = l__mod___s3_b1_conv2_bn_running_var = l__mod___s3_b1_conv2_bn_weight = l__mod___s3_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_70 = self.L__mod___s3_b1_conv2_bn_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_73 = self.L__mod___s3_b1_conv2_bn_act(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_73.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_9 = self.L__mod___s3_b1_se_fc1(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b1_se_bn = self.L__mod___s3_b1_se_bn(x_se_9);  x_se_9 = None
    x_se_10 = self.L__mod___s3_b1_se_act(l__mod___s3_b1_se_bn);  l__mod___s3_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_11 = self.L__mod___s3_b1_se_fc2(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = x_se_11.sigmoid();  x_se_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_74 = x_73 * sigmoid_2;  x_73 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_75 = self.L__mod___s3_b1_conv3_conv(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b1_conv3_bn_num_batches_tracked = self.L__mod___s3_b1_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = l__mod___s3_b1_conv3_bn_num_batches_tracked.add_(1);  l__mod___s3_b1_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv3_bn_running_mean = self.L__mod___s3_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_conv3_bn_running_var = self.L__mod___s3_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_conv3_bn_weight = self.L__mod___s3_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_conv3_bn_bias = self.L__mod___s3_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_76 = torch.nn.functional.batch_norm(x_75, l__mod___s3_b1_conv3_bn_running_mean, l__mod___s3_b1_conv3_bn_running_var, l__mod___s3_b1_conv3_bn_weight, l__mod___s3_b1_conv3_bn_bias, True, 0.1, 1e-05);  x_75 = l__mod___s3_b1_conv3_bn_running_mean = l__mod___s3_b1_conv3_bn_running_var = l__mod___s3_b1_conv3_bn_weight = l__mod___s3_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_77 = self.L__mod___s3_b1_conv3_bn_drop(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_80 = self.L__mod___s3_b1_conv3_bn_act(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b1_drop_path = self.L__mod___s3_b1_drop_path(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_81 = self.L__mod___s3_b1_downsample_conv(shortcut_2);  shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b1_downsample_bn_num_batches_tracked = self.L__mod___s3_b1_downsample_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = l__mod___s3_b1_downsample_bn_num_batches_tracked.add_(1);  l__mod___s3_b1_downsample_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_downsample_bn_running_mean = self.L__mod___s3_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b1_downsample_bn_running_var = self.L__mod___s3_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b1_downsample_bn_weight = self.L__mod___s3_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b1_downsample_bn_bias = self.L__mod___s3_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_82 = torch.nn.functional.batch_norm(x_81, l__mod___s3_b1_downsample_bn_running_mean, l__mod___s3_b1_downsample_bn_running_var, l__mod___s3_b1_downsample_bn_weight, l__mod___s3_b1_downsample_bn_bias, True, 0.1, 1e-05);  x_81 = l__mod___s3_b1_downsample_bn_running_mean = l__mod___s3_b1_downsample_bn_running_var = l__mod___s3_b1_downsample_bn_weight = l__mod___s3_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_83 = self.L__mod___s3_b1_downsample_bn_drop(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_85 = self.L__mod___s3_b1_downsample_bn_act(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_86 = l__mod___s3_b1_drop_path + x_85;  l__mod___s3_b1_drop_path = x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_3 = self.L__mod___s3_b1_act3(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_89 = self.L__mod___s3_b2_conv1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b2_conv1_bn_num_batches_tracked = self.L__mod___s3_b2_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = l__mod___s3_b2_conv1_bn_num_batches_tracked.add_(1);  l__mod___s3_b2_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv1_bn_running_mean = self.L__mod___s3_b2_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv1_bn_running_var = self.L__mod___s3_b2_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv1_bn_weight = self.L__mod___s3_b2_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv1_bn_bias = self.L__mod___s3_b2_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, l__mod___s3_b2_conv1_bn_running_mean, l__mod___s3_b2_conv1_bn_running_var, l__mod___s3_b2_conv1_bn_weight, l__mod___s3_b2_conv1_bn_bias, True, 0.1, 1e-05);  x_89 = l__mod___s3_b2_conv1_bn_running_mean = l__mod___s3_b2_conv1_bn_running_var = l__mod___s3_b2_conv1_bn_weight = l__mod___s3_b2_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_91 = self.L__mod___s3_b2_conv1_bn_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_94 = self.L__mod___s3_b2_conv1_bn_act(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_95 = self.L__mod___s3_b2_conv2_conv(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b2_conv2_bn_num_batches_tracked = self.L__mod___s3_b2_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = l__mod___s3_b2_conv2_bn_num_batches_tracked.add_(1);  l__mod___s3_b2_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv2_bn_running_mean = self.L__mod___s3_b2_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv2_bn_running_var = self.L__mod___s3_b2_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv2_bn_weight = self.L__mod___s3_b2_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv2_bn_bias = self.L__mod___s3_b2_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_96 = torch.nn.functional.batch_norm(x_95, l__mod___s3_b2_conv2_bn_running_mean, l__mod___s3_b2_conv2_bn_running_var, l__mod___s3_b2_conv2_bn_weight, l__mod___s3_b2_conv2_bn_bias, True, 0.1, 1e-05);  x_95 = l__mod___s3_b2_conv2_bn_running_mean = l__mod___s3_b2_conv2_bn_running_var = l__mod___s3_b2_conv2_bn_weight = l__mod___s3_b2_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_97 = self.L__mod___s3_b2_conv2_bn_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.L__mod___s3_b2_conv2_bn_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_100.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_13 = self.L__mod___s3_b2_se_fc1(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b2_se_bn = self.L__mod___s3_b2_se_bn(x_se_13);  x_se_13 = None
    x_se_14 = self.L__mod___s3_b2_se_act(l__mod___s3_b2_se_bn);  l__mod___s3_b2_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_15 = self.L__mod___s3_b2_se_fc2(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = x_se_15.sigmoid();  x_se_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_101 = x_100 * sigmoid_3;  x_100 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_102 = self.L__mod___s3_b2_conv3_conv(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b2_conv3_bn_num_batches_tracked = self.L__mod___s3_b2_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = l__mod___s3_b2_conv3_bn_num_batches_tracked.add_(1);  l__mod___s3_b2_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv3_bn_running_mean = self.L__mod___s3_b2_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b2_conv3_bn_running_var = self.L__mod___s3_b2_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b2_conv3_bn_weight = self.L__mod___s3_b2_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b2_conv3_bn_bias = self.L__mod___s3_b2_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_103 = torch.nn.functional.batch_norm(x_102, l__mod___s3_b2_conv3_bn_running_mean, l__mod___s3_b2_conv3_bn_running_var, l__mod___s3_b2_conv3_bn_weight, l__mod___s3_b2_conv3_bn_bias, True, 0.1, 1e-05);  x_102 = l__mod___s3_b2_conv3_bn_running_mean = l__mod___s3_b2_conv3_bn_running_var = l__mod___s3_b2_conv3_bn_weight = l__mod___s3_b2_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_104 = self.L__mod___s3_b2_conv3_bn_drop(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_107 = self.L__mod___s3_b2_conv3_bn_act(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b2_drop_path = self.L__mod___s3_b2_drop_path(x_107);  x_107 = None
    l__mod___s3_b2_downsample = self.L__mod___s3_b2_downsample(shortcut_3);  shortcut_3 = None
    x_108 = l__mod___s3_b2_drop_path + l__mod___s3_b2_downsample;  l__mod___s3_b2_drop_path = l__mod___s3_b2_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_4 = self.L__mod___s3_b2_act3(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_111 = self.L__mod___s3_b3_conv1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b3_conv1_bn_num_batches_tracked = self.L__mod___s3_b3_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = l__mod___s3_b3_conv1_bn_num_batches_tracked.add_(1);  l__mod___s3_b3_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv1_bn_running_mean = self.L__mod___s3_b3_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv1_bn_running_var = self.L__mod___s3_b3_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv1_bn_weight = self.L__mod___s3_b3_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv1_bn_bias = self.L__mod___s3_b3_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_112 = torch.nn.functional.batch_norm(x_111, l__mod___s3_b3_conv1_bn_running_mean, l__mod___s3_b3_conv1_bn_running_var, l__mod___s3_b3_conv1_bn_weight, l__mod___s3_b3_conv1_bn_bias, True, 0.1, 1e-05);  x_111 = l__mod___s3_b3_conv1_bn_running_mean = l__mod___s3_b3_conv1_bn_running_var = l__mod___s3_b3_conv1_bn_weight = l__mod___s3_b3_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_113 = self.L__mod___s3_b3_conv1_bn_drop(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_116 = self.L__mod___s3_b3_conv1_bn_act(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_117 = self.L__mod___s3_b3_conv2_conv(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b3_conv2_bn_num_batches_tracked = self.L__mod___s3_b3_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = l__mod___s3_b3_conv2_bn_num_batches_tracked.add_(1);  l__mod___s3_b3_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv2_bn_running_mean = self.L__mod___s3_b3_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv2_bn_running_var = self.L__mod___s3_b3_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv2_bn_weight = self.L__mod___s3_b3_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv2_bn_bias = self.L__mod___s3_b3_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_118 = torch.nn.functional.batch_norm(x_117, l__mod___s3_b3_conv2_bn_running_mean, l__mod___s3_b3_conv2_bn_running_var, l__mod___s3_b3_conv2_bn_weight, l__mod___s3_b3_conv2_bn_bias, True, 0.1, 1e-05);  x_117 = l__mod___s3_b3_conv2_bn_running_mean = l__mod___s3_b3_conv2_bn_running_var = l__mod___s3_b3_conv2_bn_weight = l__mod___s3_b3_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_119 = self.L__mod___s3_b3_conv2_bn_drop(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_122 = self.L__mod___s3_b3_conv2_bn_act(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_122.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_17 = self.L__mod___s3_b3_se_fc1(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b3_se_bn = self.L__mod___s3_b3_se_bn(x_se_17);  x_se_17 = None
    x_se_18 = self.L__mod___s3_b3_se_act(l__mod___s3_b3_se_bn);  l__mod___s3_b3_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_19 = self.L__mod___s3_b3_se_fc2(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = x_se_19.sigmoid();  x_se_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_123 = x_122 * sigmoid_4;  x_122 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_124 = self.L__mod___s3_b3_conv3_conv(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b3_conv3_bn_num_batches_tracked = self.L__mod___s3_b3_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = l__mod___s3_b3_conv3_bn_num_batches_tracked.add_(1);  l__mod___s3_b3_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv3_bn_running_mean = self.L__mod___s3_b3_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b3_conv3_bn_running_var = self.L__mod___s3_b3_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b3_conv3_bn_weight = self.L__mod___s3_b3_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b3_conv3_bn_bias = self.L__mod___s3_b3_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_125 = torch.nn.functional.batch_norm(x_124, l__mod___s3_b3_conv3_bn_running_mean, l__mod___s3_b3_conv3_bn_running_var, l__mod___s3_b3_conv3_bn_weight, l__mod___s3_b3_conv3_bn_bias, True, 0.1, 1e-05);  x_124 = l__mod___s3_b3_conv3_bn_running_mean = l__mod___s3_b3_conv3_bn_running_var = l__mod___s3_b3_conv3_bn_weight = l__mod___s3_b3_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_126 = self.L__mod___s3_b3_conv3_bn_drop(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_129 = self.L__mod___s3_b3_conv3_bn_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b3_drop_path = self.L__mod___s3_b3_drop_path(x_129);  x_129 = None
    l__mod___s3_b3_downsample = self.L__mod___s3_b3_downsample(shortcut_4);  shortcut_4 = None
    x_130 = l__mod___s3_b3_drop_path + l__mod___s3_b3_downsample;  l__mod___s3_b3_drop_path = l__mod___s3_b3_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_5 = self.L__mod___s3_b3_act3(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_133 = self.L__mod___s3_b4_conv1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b4_conv1_bn_num_batches_tracked = self.L__mod___s3_b4_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = l__mod___s3_b4_conv1_bn_num_batches_tracked.add_(1);  l__mod___s3_b4_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv1_bn_running_mean = self.L__mod___s3_b4_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv1_bn_running_var = self.L__mod___s3_b4_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv1_bn_weight = self.L__mod___s3_b4_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv1_bn_bias = self.L__mod___s3_b4_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, l__mod___s3_b4_conv1_bn_running_mean, l__mod___s3_b4_conv1_bn_running_var, l__mod___s3_b4_conv1_bn_weight, l__mod___s3_b4_conv1_bn_bias, True, 0.1, 1e-05);  x_133 = l__mod___s3_b4_conv1_bn_running_mean = l__mod___s3_b4_conv1_bn_running_var = l__mod___s3_b4_conv1_bn_weight = l__mod___s3_b4_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.L__mod___s3_b4_conv1_bn_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.L__mod___s3_b4_conv1_bn_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_139 = self.L__mod___s3_b4_conv2_conv(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b4_conv2_bn_num_batches_tracked = self.L__mod___s3_b4_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = l__mod___s3_b4_conv2_bn_num_batches_tracked.add_(1);  l__mod___s3_b4_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv2_bn_running_mean = self.L__mod___s3_b4_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv2_bn_running_var = self.L__mod___s3_b4_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv2_bn_weight = self.L__mod___s3_b4_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv2_bn_bias = self.L__mod___s3_b4_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_139, l__mod___s3_b4_conv2_bn_running_mean, l__mod___s3_b4_conv2_bn_running_var, l__mod___s3_b4_conv2_bn_weight, l__mod___s3_b4_conv2_bn_bias, True, 0.1, 1e-05);  x_139 = l__mod___s3_b4_conv2_bn_running_mean = l__mod___s3_b4_conv2_bn_running_var = l__mod___s3_b4_conv2_bn_weight = l__mod___s3_b4_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.L__mod___s3_b4_conv2_bn_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_144 = self.L__mod___s3_b4_conv2_bn_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_144.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_21 = self.L__mod___s3_b4_se_fc1(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s3_b4_se_bn = self.L__mod___s3_b4_se_bn(x_se_21);  x_se_21 = None
    x_se_22 = self.L__mod___s3_b4_se_act(l__mod___s3_b4_se_bn);  l__mod___s3_b4_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_23 = self.L__mod___s3_b4_se_fc2(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5 = x_se_23.sigmoid();  x_se_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_145 = x_144 * sigmoid_5;  x_144 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_146 = self.L__mod___s3_b4_conv3_conv(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s3_b4_conv3_bn_num_batches_tracked = self.L__mod___s3_b4_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = l__mod___s3_b4_conv3_bn_num_batches_tracked.add_(1);  l__mod___s3_b4_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv3_bn_running_mean = self.L__mod___s3_b4_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s3_b4_conv3_bn_running_var = self.L__mod___s3_b4_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s3_b4_conv3_bn_weight = self.L__mod___s3_b4_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s3_b4_conv3_bn_bias = self.L__mod___s3_b4_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_147 = torch.nn.functional.batch_norm(x_146, l__mod___s3_b4_conv3_bn_running_mean, l__mod___s3_b4_conv3_bn_running_var, l__mod___s3_b4_conv3_bn_weight, l__mod___s3_b4_conv3_bn_bias, True, 0.1, 1e-05);  x_146 = l__mod___s3_b4_conv3_bn_running_mean = l__mod___s3_b4_conv3_bn_running_var = l__mod___s3_b4_conv3_bn_weight = l__mod___s3_b4_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_148 = self.L__mod___s3_b4_conv3_bn_drop(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_151 = self.L__mod___s3_b4_conv3_bn_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s3_b4_drop_path = self.L__mod___s3_b4_drop_path(x_151);  x_151 = None
    l__mod___s3_b4_downsample = self.L__mod___s3_b4_downsample(shortcut_5);  shortcut_5 = None
    x_152 = l__mod___s3_b4_drop_path + l__mod___s3_b4_downsample;  l__mod___s3_b4_drop_path = l__mod___s3_b4_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_6 = self.L__mod___s3_b4_act3(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_156 = self.L__mod___s4_b1_conv1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b1_conv1_bn_num_batches_tracked = self.L__mod___s4_b1_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = l__mod___s4_b1_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b1_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv1_bn_running_mean = self.L__mod___s4_b1_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv1_bn_running_var = self.L__mod___s4_b1_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv1_bn_weight = self.L__mod___s4_b1_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv1_bn_bias = self.L__mod___s4_b1_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_157 = torch.nn.functional.batch_norm(x_156, l__mod___s4_b1_conv1_bn_running_mean, l__mod___s4_b1_conv1_bn_running_var, l__mod___s4_b1_conv1_bn_weight, l__mod___s4_b1_conv1_bn_bias, True, 0.1, 1e-05);  x_156 = l__mod___s4_b1_conv1_bn_running_mean = l__mod___s4_b1_conv1_bn_running_var = l__mod___s4_b1_conv1_bn_weight = l__mod___s4_b1_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_158 = self.L__mod___s4_b1_conv1_bn_drop(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_161 = self.L__mod___s4_b1_conv1_bn_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_162 = self.L__mod___s4_b1_conv2_conv(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b1_conv2_bn_num_batches_tracked = self.L__mod___s4_b1_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = l__mod___s4_b1_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b1_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv2_bn_running_mean = self.L__mod___s4_b1_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv2_bn_running_var = self.L__mod___s4_b1_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv2_bn_weight = self.L__mod___s4_b1_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv2_bn_bias = self.L__mod___s4_b1_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_163 = torch.nn.functional.batch_norm(x_162, l__mod___s4_b1_conv2_bn_running_mean, l__mod___s4_b1_conv2_bn_running_var, l__mod___s4_b1_conv2_bn_weight, l__mod___s4_b1_conv2_bn_bias, True, 0.1, 1e-05);  x_162 = l__mod___s4_b1_conv2_bn_running_mean = l__mod___s4_b1_conv2_bn_running_var = l__mod___s4_b1_conv2_bn_weight = l__mod___s4_b1_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_164 = self.L__mod___s4_b1_conv2_bn_drop(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_167 = self.L__mod___s4_b1_conv2_bn_act(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_167.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_25 = self.L__mod___s4_b1_se_fc1(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b1_se_bn = self.L__mod___s4_b1_se_bn(x_se_25);  x_se_25 = None
    x_se_26 = self.L__mod___s4_b1_se_act(l__mod___s4_b1_se_bn);  l__mod___s4_b1_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_27 = self.L__mod___s4_b1_se_fc2(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6 = x_se_27.sigmoid();  x_se_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_168 = x_167 * sigmoid_6;  x_167 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_169 = self.L__mod___s4_b1_conv3_conv(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b1_conv3_bn_num_batches_tracked = self.L__mod___s4_b1_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = l__mod___s4_b1_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b1_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv3_bn_running_mean = self.L__mod___s4_b1_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_conv3_bn_running_var = self.L__mod___s4_b1_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_conv3_bn_weight = self.L__mod___s4_b1_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_conv3_bn_bias = self.L__mod___s4_b1_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_170 = torch.nn.functional.batch_norm(x_169, l__mod___s4_b1_conv3_bn_running_mean, l__mod___s4_b1_conv3_bn_running_var, l__mod___s4_b1_conv3_bn_weight, l__mod___s4_b1_conv3_bn_bias, True, 0.1, 1e-05);  x_169 = l__mod___s4_b1_conv3_bn_running_mean = l__mod___s4_b1_conv3_bn_running_var = l__mod___s4_b1_conv3_bn_weight = l__mod___s4_b1_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_171 = self.L__mod___s4_b1_conv3_bn_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_174 = self.L__mod___s4_b1_conv3_bn_act(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b1_drop_path = self.L__mod___s4_b1_drop_path(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_175 = self.L__mod___s4_b1_downsample_conv(shortcut_6);  shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b1_downsample_bn_num_batches_tracked = self.L__mod___s4_b1_downsample_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = l__mod___s4_b1_downsample_bn_num_batches_tracked.add_(1);  l__mod___s4_b1_downsample_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_downsample_bn_running_mean = self.L__mod___s4_b1_downsample_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b1_downsample_bn_running_var = self.L__mod___s4_b1_downsample_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b1_downsample_bn_weight = self.L__mod___s4_b1_downsample_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b1_downsample_bn_bias = self.L__mod___s4_b1_downsample_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_176 = torch.nn.functional.batch_norm(x_175, l__mod___s4_b1_downsample_bn_running_mean, l__mod___s4_b1_downsample_bn_running_var, l__mod___s4_b1_downsample_bn_weight, l__mod___s4_b1_downsample_bn_bias, True, 0.1, 1e-05);  x_175 = l__mod___s4_b1_downsample_bn_running_mean = l__mod___s4_b1_downsample_bn_running_var = l__mod___s4_b1_downsample_bn_weight = l__mod___s4_b1_downsample_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_177 = self.L__mod___s4_b1_downsample_bn_drop(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_179 = self.L__mod___s4_b1_downsample_bn_act(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    x_180 = l__mod___s4_b1_drop_path + x_179;  l__mod___s4_b1_drop_path = x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_7 = self.L__mod___s4_b1_act3(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_183 = self.L__mod___s4_b2_conv1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b2_conv1_bn_num_batches_tracked = self.L__mod___s4_b2_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = l__mod___s4_b2_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b2_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv1_bn_running_mean = self.L__mod___s4_b2_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv1_bn_running_var = self.L__mod___s4_b2_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b2_conv1_bn_weight = self.L__mod___s4_b2_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b2_conv1_bn_bias = self.L__mod___s4_b2_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_184 = torch.nn.functional.batch_norm(x_183, l__mod___s4_b2_conv1_bn_running_mean, l__mod___s4_b2_conv1_bn_running_var, l__mod___s4_b2_conv1_bn_weight, l__mod___s4_b2_conv1_bn_bias, True, 0.1, 1e-05);  x_183 = l__mod___s4_b2_conv1_bn_running_mean = l__mod___s4_b2_conv1_bn_running_var = l__mod___s4_b2_conv1_bn_weight = l__mod___s4_b2_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_185 = self.L__mod___s4_b2_conv1_bn_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_188 = self.L__mod___s4_b2_conv1_bn_act(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_189 = self.L__mod___s4_b2_conv2_conv(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b2_conv2_bn_num_batches_tracked = self.L__mod___s4_b2_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = l__mod___s4_b2_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b2_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv2_bn_running_mean = self.L__mod___s4_b2_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv2_bn_running_var = self.L__mod___s4_b2_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b2_conv2_bn_weight = self.L__mod___s4_b2_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b2_conv2_bn_bias = self.L__mod___s4_b2_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_190 = torch.nn.functional.batch_norm(x_189, l__mod___s4_b2_conv2_bn_running_mean, l__mod___s4_b2_conv2_bn_running_var, l__mod___s4_b2_conv2_bn_weight, l__mod___s4_b2_conv2_bn_bias, True, 0.1, 1e-05);  x_189 = l__mod___s4_b2_conv2_bn_running_mean = l__mod___s4_b2_conv2_bn_running_var = l__mod___s4_b2_conv2_bn_weight = l__mod___s4_b2_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_191 = self.L__mod___s4_b2_conv2_bn_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_194 = self.L__mod___s4_b2_conv2_bn_act(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_194.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_29 = self.L__mod___s4_b2_se_fc1(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b2_se_bn = self.L__mod___s4_b2_se_bn(x_se_29);  x_se_29 = None
    x_se_30 = self.L__mod___s4_b2_se_act(l__mod___s4_b2_se_bn);  l__mod___s4_b2_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_31 = self.L__mod___s4_b2_se_fc2(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7 = x_se_31.sigmoid();  x_se_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_195 = x_194 * sigmoid_7;  x_194 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_196 = self.L__mod___s4_b2_conv3_conv(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b2_conv3_bn_num_batches_tracked = self.L__mod___s4_b2_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = l__mod___s4_b2_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b2_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv3_bn_running_mean = self.L__mod___s4_b2_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b2_conv3_bn_running_var = self.L__mod___s4_b2_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b2_conv3_bn_weight = self.L__mod___s4_b2_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b2_conv3_bn_bias = self.L__mod___s4_b2_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_197 = torch.nn.functional.batch_norm(x_196, l__mod___s4_b2_conv3_bn_running_mean, l__mod___s4_b2_conv3_bn_running_var, l__mod___s4_b2_conv3_bn_weight, l__mod___s4_b2_conv3_bn_bias, True, 0.1, 1e-05);  x_196 = l__mod___s4_b2_conv3_bn_running_mean = l__mod___s4_b2_conv3_bn_running_var = l__mod___s4_b2_conv3_bn_weight = l__mod___s4_b2_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.L__mod___s4_b2_conv3_bn_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_201 = self.L__mod___s4_b2_conv3_bn_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b2_drop_path = self.L__mod___s4_b2_drop_path(x_201);  x_201 = None
    l__mod___s4_b2_downsample = self.L__mod___s4_b2_downsample(shortcut_7);  shortcut_7 = None
    x_202 = l__mod___s4_b2_drop_path + l__mod___s4_b2_downsample;  l__mod___s4_b2_drop_path = l__mod___s4_b2_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_8 = self.L__mod___s4_b2_act3(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_205 = self.L__mod___s4_b3_conv1_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b3_conv1_bn_num_batches_tracked = self.L__mod___s4_b3_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = l__mod___s4_b3_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b3_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv1_bn_running_mean = self.L__mod___s4_b3_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv1_bn_running_var = self.L__mod___s4_b3_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b3_conv1_bn_weight = self.L__mod___s4_b3_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b3_conv1_bn_bias = self.L__mod___s4_b3_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_205, l__mod___s4_b3_conv1_bn_running_mean, l__mod___s4_b3_conv1_bn_running_var, l__mod___s4_b3_conv1_bn_weight, l__mod___s4_b3_conv1_bn_bias, True, 0.1, 1e-05);  x_205 = l__mod___s4_b3_conv1_bn_running_mean = l__mod___s4_b3_conv1_bn_running_var = l__mod___s4_b3_conv1_bn_weight = l__mod___s4_b3_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.L__mod___s4_b3_conv1_bn_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_210 = self.L__mod___s4_b3_conv1_bn_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_211 = self.L__mod___s4_b3_conv2_conv(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b3_conv2_bn_num_batches_tracked = self.L__mod___s4_b3_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = l__mod___s4_b3_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b3_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv2_bn_running_mean = self.L__mod___s4_b3_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv2_bn_running_var = self.L__mod___s4_b3_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b3_conv2_bn_weight = self.L__mod___s4_b3_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b3_conv2_bn_bias = self.L__mod___s4_b3_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_212 = torch.nn.functional.batch_norm(x_211, l__mod___s4_b3_conv2_bn_running_mean, l__mod___s4_b3_conv2_bn_running_var, l__mod___s4_b3_conv2_bn_weight, l__mod___s4_b3_conv2_bn_bias, True, 0.1, 1e-05);  x_211 = l__mod___s4_b3_conv2_bn_running_mean = l__mod___s4_b3_conv2_bn_running_var = l__mod___s4_b3_conv2_bn_weight = l__mod___s4_b3_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_213 = self.L__mod___s4_b3_conv2_bn_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_216 = self.L__mod___s4_b3_conv2_bn_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_216.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_33 = self.L__mod___s4_b3_se_fc1(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b3_se_bn = self.L__mod___s4_b3_se_bn(x_se_33);  x_se_33 = None
    x_se_34 = self.L__mod___s4_b3_se_act(l__mod___s4_b3_se_bn);  l__mod___s4_b3_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_35 = self.L__mod___s4_b3_se_fc2(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8 = x_se_35.sigmoid();  x_se_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_217 = x_216 * sigmoid_8;  x_216 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_218 = self.L__mod___s4_b3_conv3_conv(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b3_conv3_bn_num_batches_tracked = self.L__mod___s4_b3_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = l__mod___s4_b3_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b3_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv3_bn_running_mean = self.L__mod___s4_b3_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b3_conv3_bn_running_var = self.L__mod___s4_b3_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b3_conv3_bn_weight = self.L__mod___s4_b3_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b3_conv3_bn_bias = self.L__mod___s4_b3_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_219 = torch.nn.functional.batch_norm(x_218, l__mod___s4_b3_conv3_bn_running_mean, l__mod___s4_b3_conv3_bn_running_var, l__mod___s4_b3_conv3_bn_weight, l__mod___s4_b3_conv3_bn_bias, True, 0.1, 1e-05);  x_218 = l__mod___s4_b3_conv3_bn_running_mean = l__mod___s4_b3_conv3_bn_running_var = l__mod___s4_b3_conv3_bn_weight = l__mod___s4_b3_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_220 = self.L__mod___s4_b3_conv3_bn_drop(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_223 = self.L__mod___s4_b3_conv3_bn_act(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b3_drop_path = self.L__mod___s4_b3_drop_path(x_223);  x_223 = None
    l__mod___s4_b3_downsample = self.L__mod___s4_b3_downsample(shortcut_8);  shortcut_8 = None
    x_224 = l__mod___s4_b3_drop_path + l__mod___s4_b3_downsample;  l__mod___s4_b3_drop_path = l__mod___s4_b3_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_9 = self.L__mod___s4_b3_act3(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.L__mod___s4_b4_conv1_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b4_conv1_bn_num_batches_tracked = self.L__mod___s4_b4_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__32 = l__mod___s4_b4_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b4_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv1_bn_running_mean = self.L__mod___s4_b4_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv1_bn_running_var = self.L__mod___s4_b4_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b4_conv1_bn_weight = self.L__mod___s4_b4_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b4_conv1_bn_bias = self.L__mod___s4_b4_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, l__mod___s4_b4_conv1_bn_running_mean, l__mod___s4_b4_conv1_bn_running_var, l__mod___s4_b4_conv1_bn_weight, l__mod___s4_b4_conv1_bn_bias, True, 0.1, 1e-05);  x_227 = l__mod___s4_b4_conv1_bn_running_mean = l__mod___s4_b4_conv1_bn_running_var = l__mod___s4_b4_conv1_bn_weight = l__mod___s4_b4_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.L__mod___s4_b4_conv1_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_232 = self.L__mod___s4_b4_conv1_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_233 = self.L__mod___s4_b4_conv2_conv(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b4_conv2_bn_num_batches_tracked = self.L__mod___s4_b4_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__33 = l__mod___s4_b4_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b4_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv2_bn_running_mean = self.L__mod___s4_b4_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv2_bn_running_var = self.L__mod___s4_b4_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b4_conv2_bn_weight = self.L__mod___s4_b4_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b4_conv2_bn_bias = self.L__mod___s4_b4_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_234 = torch.nn.functional.batch_norm(x_233, l__mod___s4_b4_conv2_bn_running_mean, l__mod___s4_b4_conv2_bn_running_var, l__mod___s4_b4_conv2_bn_weight, l__mod___s4_b4_conv2_bn_bias, True, 0.1, 1e-05);  x_233 = l__mod___s4_b4_conv2_bn_running_mean = l__mod___s4_b4_conv2_bn_running_var = l__mod___s4_b4_conv2_bn_weight = l__mod___s4_b4_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_235 = self.L__mod___s4_b4_conv2_bn_drop(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_238 = self.L__mod___s4_b4_conv2_bn_act(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_238.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_37 = self.L__mod___s4_b4_se_fc1(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b4_se_bn = self.L__mod___s4_b4_se_bn(x_se_37);  x_se_37 = None
    x_se_38 = self.L__mod___s4_b4_se_act(l__mod___s4_b4_se_bn);  l__mod___s4_b4_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_39 = self.L__mod___s4_b4_se_fc2(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9 = x_se_39.sigmoid();  x_se_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_239 = x_238 * sigmoid_9;  x_238 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_240 = self.L__mod___s4_b4_conv3_conv(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b4_conv3_bn_num_batches_tracked = self.L__mod___s4_b4_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__34 = l__mod___s4_b4_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b4_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv3_bn_running_mean = self.L__mod___s4_b4_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b4_conv3_bn_running_var = self.L__mod___s4_b4_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b4_conv3_bn_weight = self.L__mod___s4_b4_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b4_conv3_bn_bias = self.L__mod___s4_b4_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_241 = torch.nn.functional.batch_norm(x_240, l__mod___s4_b4_conv3_bn_running_mean, l__mod___s4_b4_conv3_bn_running_var, l__mod___s4_b4_conv3_bn_weight, l__mod___s4_b4_conv3_bn_bias, True, 0.1, 1e-05);  x_240 = l__mod___s4_b4_conv3_bn_running_mean = l__mod___s4_b4_conv3_bn_running_var = l__mod___s4_b4_conv3_bn_weight = l__mod___s4_b4_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_242 = self.L__mod___s4_b4_conv3_bn_drop(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_245 = self.L__mod___s4_b4_conv3_bn_act(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b4_drop_path = self.L__mod___s4_b4_drop_path(x_245);  x_245 = None
    l__mod___s4_b4_downsample = self.L__mod___s4_b4_downsample(shortcut_9);  shortcut_9 = None
    x_246 = l__mod___s4_b4_drop_path + l__mod___s4_b4_downsample;  l__mod___s4_b4_drop_path = l__mod___s4_b4_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_10 = self.L__mod___s4_b4_act3(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_249 = self.L__mod___s4_b5_conv1_conv(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b5_conv1_bn_num_batches_tracked = self.L__mod___s4_b5_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__35 = l__mod___s4_b5_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b5_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv1_bn_running_mean = self.L__mod___s4_b5_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv1_bn_running_var = self.L__mod___s4_b5_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b5_conv1_bn_weight = self.L__mod___s4_b5_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b5_conv1_bn_bias = self.L__mod___s4_b5_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_250 = torch.nn.functional.batch_norm(x_249, l__mod___s4_b5_conv1_bn_running_mean, l__mod___s4_b5_conv1_bn_running_var, l__mod___s4_b5_conv1_bn_weight, l__mod___s4_b5_conv1_bn_bias, True, 0.1, 1e-05);  x_249 = l__mod___s4_b5_conv1_bn_running_mean = l__mod___s4_b5_conv1_bn_running_var = l__mod___s4_b5_conv1_bn_weight = l__mod___s4_b5_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_251 = self.L__mod___s4_b5_conv1_bn_drop(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_254 = self.L__mod___s4_b5_conv1_bn_act(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_255 = self.L__mod___s4_b5_conv2_conv(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b5_conv2_bn_num_batches_tracked = self.L__mod___s4_b5_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__36 = l__mod___s4_b5_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b5_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv2_bn_running_mean = self.L__mod___s4_b5_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv2_bn_running_var = self.L__mod___s4_b5_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b5_conv2_bn_weight = self.L__mod___s4_b5_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b5_conv2_bn_bias = self.L__mod___s4_b5_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_256 = torch.nn.functional.batch_norm(x_255, l__mod___s4_b5_conv2_bn_running_mean, l__mod___s4_b5_conv2_bn_running_var, l__mod___s4_b5_conv2_bn_weight, l__mod___s4_b5_conv2_bn_bias, True, 0.1, 1e-05);  x_255 = l__mod___s4_b5_conv2_bn_running_mean = l__mod___s4_b5_conv2_bn_running_var = l__mod___s4_b5_conv2_bn_weight = l__mod___s4_b5_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_257 = self.L__mod___s4_b5_conv2_bn_drop(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_260 = self.L__mod___s4_b5_conv2_bn_act(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_260.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_41 = self.L__mod___s4_b5_se_fc1(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b5_se_bn = self.L__mod___s4_b5_se_bn(x_se_41);  x_se_41 = None
    x_se_42 = self.L__mod___s4_b5_se_act(l__mod___s4_b5_se_bn);  l__mod___s4_b5_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_43 = self.L__mod___s4_b5_se_fc2(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10 = x_se_43.sigmoid();  x_se_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_261 = x_260 * sigmoid_10;  x_260 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_262 = self.L__mod___s4_b5_conv3_conv(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b5_conv3_bn_num_batches_tracked = self.L__mod___s4_b5_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__37 = l__mod___s4_b5_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b5_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv3_bn_running_mean = self.L__mod___s4_b5_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b5_conv3_bn_running_var = self.L__mod___s4_b5_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b5_conv3_bn_weight = self.L__mod___s4_b5_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b5_conv3_bn_bias = self.L__mod___s4_b5_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_263 = torch.nn.functional.batch_norm(x_262, l__mod___s4_b5_conv3_bn_running_mean, l__mod___s4_b5_conv3_bn_running_var, l__mod___s4_b5_conv3_bn_weight, l__mod___s4_b5_conv3_bn_bias, True, 0.1, 1e-05);  x_262 = l__mod___s4_b5_conv3_bn_running_mean = l__mod___s4_b5_conv3_bn_running_var = l__mod___s4_b5_conv3_bn_weight = l__mod___s4_b5_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_264 = self.L__mod___s4_b5_conv3_bn_drop(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_267 = self.L__mod___s4_b5_conv3_bn_act(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b5_drop_path = self.L__mod___s4_b5_drop_path(x_267);  x_267 = None
    l__mod___s4_b5_downsample = self.L__mod___s4_b5_downsample(shortcut_10);  shortcut_10 = None
    x_268 = l__mod___s4_b5_drop_path + l__mod___s4_b5_downsample;  l__mod___s4_b5_drop_path = l__mod___s4_b5_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_11 = self.L__mod___s4_b5_act3(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_271 = self.L__mod___s4_b6_conv1_conv(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b6_conv1_bn_num_batches_tracked = self.L__mod___s4_b6_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__38 = l__mod___s4_b6_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b6_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv1_bn_running_mean = self.L__mod___s4_b6_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv1_bn_running_var = self.L__mod___s4_b6_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b6_conv1_bn_weight = self.L__mod___s4_b6_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b6_conv1_bn_bias = self.L__mod___s4_b6_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_272 = torch.nn.functional.batch_norm(x_271, l__mod___s4_b6_conv1_bn_running_mean, l__mod___s4_b6_conv1_bn_running_var, l__mod___s4_b6_conv1_bn_weight, l__mod___s4_b6_conv1_bn_bias, True, 0.1, 1e-05);  x_271 = l__mod___s4_b6_conv1_bn_running_mean = l__mod___s4_b6_conv1_bn_running_var = l__mod___s4_b6_conv1_bn_weight = l__mod___s4_b6_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_273 = self.L__mod___s4_b6_conv1_bn_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_276 = self.L__mod___s4_b6_conv1_bn_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_277 = self.L__mod___s4_b6_conv2_conv(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b6_conv2_bn_num_batches_tracked = self.L__mod___s4_b6_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__39 = l__mod___s4_b6_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b6_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv2_bn_running_mean = self.L__mod___s4_b6_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv2_bn_running_var = self.L__mod___s4_b6_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b6_conv2_bn_weight = self.L__mod___s4_b6_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b6_conv2_bn_bias = self.L__mod___s4_b6_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_278 = torch.nn.functional.batch_norm(x_277, l__mod___s4_b6_conv2_bn_running_mean, l__mod___s4_b6_conv2_bn_running_var, l__mod___s4_b6_conv2_bn_weight, l__mod___s4_b6_conv2_bn_bias, True, 0.1, 1e-05);  x_277 = l__mod___s4_b6_conv2_bn_running_mean = l__mod___s4_b6_conv2_bn_running_var = l__mod___s4_b6_conv2_bn_weight = l__mod___s4_b6_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_279 = self.L__mod___s4_b6_conv2_bn_drop(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_282 = self.L__mod___s4_b6_conv2_bn_act(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_282.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_45 = self.L__mod___s4_b6_se_fc1(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b6_se_bn = self.L__mod___s4_b6_se_bn(x_se_45);  x_se_45 = None
    x_se_46 = self.L__mod___s4_b6_se_act(l__mod___s4_b6_se_bn);  l__mod___s4_b6_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_47 = self.L__mod___s4_b6_se_fc2(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11 = x_se_47.sigmoid();  x_se_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_283 = x_282 * sigmoid_11;  x_282 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_284 = self.L__mod___s4_b6_conv3_conv(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b6_conv3_bn_num_batches_tracked = self.L__mod___s4_b6_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__40 = l__mod___s4_b6_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b6_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv3_bn_running_mean = self.L__mod___s4_b6_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b6_conv3_bn_running_var = self.L__mod___s4_b6_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b6_conv3_bn_weight = self.L__mod___s4_b6_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b6_conv3_bn_bias = self.L__mod___s4_b6_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_285 = torch.nn.functional.batch_norm(x_284, l__mod___s4_b6_conv3_bn_running_mean, l__mod___s4_b6_conv3_bn_running_var, l__mod___s4_b6_conv3_bn_weight, l__mod___s4_b6_conv3_bn_bias, True, 0.1, 1e-05);  x_284 = l__mod___s4_b6_conv3_bn_running_mean = l__mod___s4_b6_conv3_bn_running_var = l__mod___s4_b6_conv3_bn_weight = l__mod___s4_b6_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_286 = self.L__mod___s4_b6_conv3_bn_drop(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_289 = self.L__mod___s4_b6_conv3_bn_act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b6_drop_path = self.L__mod___s4_b6_drop_path(x_289);  x_289 = None
    l__mod___s4_b6_downsample = self.L__mod___s4_b6_downsample(shortcut_11);  shortcut_11 = None
    x_290 = l__mod___s4_b6_drop_path + l__mod___s4_b6_downsample;  l__mod___s4_b6_drop_path = l__mod___s4_b6_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    shortcut_12 = self.L__mod___s4_b6_act3(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_293 = self.L__mod___s4_b7_conv1_conv(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b7_conv1_bn_num_batches_tracked = self.L__mod___s4_b7_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__41 = l__mod___s4_b7_conv1_bn_num_batches_tracked.add_(1);  l__mod___s4_b7_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv1_bn_running_mean = self.L__mod___s4_b7_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv1_bn_running_var = self.L__mod___s4_b7_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b7_conv1_bn_weight = self.L__mod___s4_b7_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b7_conv1_bn_bias = self.L__mod___s4_b7_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_294 = torch.nn.functional.batch_norm(x_293, l__mod___s4_b7_conv1_bn_running_mean, l__mod___s4_b7_conv1_bn_running_var, l__mod___s4_b7_conv1_bn_weight, l__mod___s4_b7_conv1_bn_bias, True, 0.1, 1e-05);  x_293 = l__mod___s4_b7_conv1_bn_running_mean = l__mod___s4_b7_conv1_bn_running_var = l__mod___s4_b7_conv1_bn_weight = l__mod___s4_b7_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_295 = self.L__mod___s4_b7_conv1_bn_drop(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_298 = self.L__mod___s4_b7_conv1_bn_act(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_299 = self.L__mod___s4_b7_conv2_conv(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b7_conv2_bn_num_batches_tracked = self.L__mod___s4_b7_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__42 = l__mod___s4_b7_conv2_bn_num_batches_tracked.add_(1);  l__mod___s4_b7_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv2_bn_running_mean = self.L__mod___s4_b7_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv2_bn_running_var = self.L__mod___s4_b7_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b7_conv2_bn_weight = self.L__mod___s4_b7_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b7_conv2_bn_bias = self.L__mod___s4_b7_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_300 = torch.nn.functional.batch_norm(x_299, l__mod___s4_b7_conv2_bn_running_mean, l__mod___s4_b7_conv2_bn_running_var, l__mod___s4_b7_conv2_bn_weight, l__mod___s4_b7_conv2_bn_bias, True, 0.1, 1e-05);  x_299 = l__mod___s4_b7_conv2_bn_running_mean = l__mod___s4_b7_conv2_bn_running_var = l__mod___s4_b7_conv2_bn_weight = l__mod___s4_b7_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_301 = self.L__mod___s4_b7_conv2_bn_drop(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_304 = self.L__mod___s4_b7_conv2_bn_act(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_304.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_49 = self.L__mod___s4_b7_se_fc1(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    l__mod___s4_b7_se_bn = self.L__mod___s4_b7_se_bn(x_se_49);  x_se_49 = None
    x_se_50 = self.L__mod___s4_b7_se_act(l__mod___s4_b7_se_bn);  l__mod___s4_b7_se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_51 = self.L__mod___s4_b7_se_fc2(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12 = x_se_51.sigmoid();  x_se_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_305 = x_304 * sigmoid_12;  x_304 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_306 = self.L__mod___s4_b7_conv3_conv(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___s4_b7_conv3_bn_num_batches_tracked = self.L__mod___s4_b7_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__43 = l__mod___s4_b7_conv3_bn_num_batches_tracked.add_(1);  l__mod___s4_b7_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv3_bn_running_mean = self.L__mod___s4_b7_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___s4_b7_conv3_bn_running_var = self.L__mod___s4_b7_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___s4_b7_conv3_bn_weight = self.L__mod___s4_b7_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___s4_b7_conv3_bn_bias = self.L__mod___s4_b7_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_307 = torch.nn.functional.batch_norm(x_306, l__mod___s4_b7_conv3_bn_running_mean, l__mod___s4_b7_conv3_bn_running_var, l__mod___s4_b7_conv3_bn_weight, l__mod___s4_b7_conv3_bn_bias, True, 0.1, 1e-05);  x_306 = l__mod___s4_b7_conv3_bn_running_mean = l__mod___s4_b7_conv3_bn_running_var = l__mod___s4_b7_conv3_bn_weight = l__mod___s4_b7_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_308 = self.L__mod___s4_b7_conv3_bn_drop(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_311 = self.L__mod___s4_b7_conv3_bn_act(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    l__mod___s4_b7_drop_path = self.L__mod___s4_b7_drop_path(x_311);  x_311 = None
    l__mod___s4_b7_downsample = self.L__mod___s4_b7_downsample(shortcut_12);  shortcut_12 = None
    x_312 = l__mod___s4_b7_drop_path + l__mod___s4_b7_downsample;  l__mod___s4_b7_drop_path = l__mod___s4_b7_downsample = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    x_315 = self.L__mod___s4_b7_act3(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:524, code: x = self.final_conv(x)
    x_317 = self.L__mod___final_conv(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_318 = self.L__mod___head_global_pool_pool(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_320 = self.L__mod___head_global_pool_flatten(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_321 = self.L__mod___head_drop(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_322 = self.L__mod___head_fc(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    pred = self.L__mod___head_flatten(x_322);  x_322 = None
    return (pred,)
    