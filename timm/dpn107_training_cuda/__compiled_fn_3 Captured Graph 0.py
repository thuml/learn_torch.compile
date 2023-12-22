from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___features_conv1_1_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv1_1_bn_num_batches_tracked = self.L__mod___features_conv1_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___features_conv1_1_bn_num_batches_tracked.add_(1);  l__mod___features_conv1_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv1_1_bn_running_mean = self.L__mod___features_conv1_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv1_1_bn_running_var = self.L__mod___features_conv1_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv1_1_bn_weight = self.L__mod___features_conv1_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv1_1_bn_bias = self.L__mod___features_conv1_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___features_conv1_1_bn_running_mean, l__mod___features_conv1_1_bn_running_var, l__mod___features_conv1_1_bn_weight, l__mod___features_conv1_1_bn_bias, True, 0.1, 0.001);  x = l__mod___features_conv1_1_bn_running_mean = l__mod___features_conv1_1_bn_running_var = l__mod___features_conv1_1_bn_weight = l__mod___features_conv1_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___features_conv1_1_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_4 = self.L__mod___features_conv1_1_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:266, code: return self.features(x)
    x_in = self.L__mod___features_conv1_pool(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_1_c1x1_w_s1_bn_num_batches_tracked = self.L__mod___features_conv2_1_c1x1_w_s1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = l__mod___features_conv2_1_c1x1_w_s1_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_1_c1x1_w_s1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_w_s1_bn_running_mean = self.L__mod___features_conv2_1_c1x1_w_s1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_w_s1_bn_running_var = self.L__mod___features_conv2_1_c1x1_w_s1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_1_c1x1_w_s1_bn_weight = self.L__mod___features_conv2_1_c1x1_w_s1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_1_c1x1_w_s1_bn_bias = self.L__mod___features_conv2_1_c1x1_w_s1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_5 = torch.nn.functional.batch_norm(x_in, l__mod___features_conv2_1_c1x1_w_s1_bn_running_mean, l__mod___features_conv2_1_c1x1_w_s1_bn_running_var, l__mod___features_conv2_1_c1x1_w_s1_bn_weight, l__mod___features_conv2_1_c1x1_w_s1_bn_bias, True, 0.1, 0.001);  l__mod___features_conv2_1_c1x1_w_s1_bn_running_mean = l__mod___features_conv2_1_c1x1_w_s1_bn_running_var = l__mod___features_conv2_1_c1x1_w_s1_bn_weight = l__mod___features_conv2_1_c1x1_w_s1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_6 = self.L__mod___features_conv2_1_c1x1_w_s1_bn_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_7 = self.L__mod___features_conv2_1_c1x1_w_s1_bn_act(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_s = self.L__mod___features_conv2_1_c1x1_w_s1_conv(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    x_s1 = x_s[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    x_s2 = x_s[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  x_s = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_1_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv2_1_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = l__mod___features_conv2_1_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_1_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_a_bn_running_mean = self.L__mod___features_conv2_1_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_a_bn_running_var = self.L__mod___features_conv2_1_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_1_c1x1_a_bn_weight = self.L__mod___features_conv2_1_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_1_c1x1_a_bn_bias = self.L__mod___features_conv2_1_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_8 = torch.nn.functional.batch_norm(x_in, l__mod___features_conv2_1_c1x1_a_bn_running_mean, l__mod___features_conv2_1_c1x1_a_bn_running_var, l__mod___features_conv2_1_c1x1_a_bn_weight, l__mod___features_conv2_1_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in = l__mod___features_conv2_1_c1x1_a_bn_running_mean = l__mod___features_conv2_1_c1x1_a_bn_running_var = l__mod___features_conv2_1_c1x1_a_bn_weight = l__mod___features_conv2_1_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_9 = self.L__mod___features_conv2_1_c1x1_a_bn_drop(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_10 = self.L__mod___features_conv2_1_c1x1_a_bn_act(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_1 = self.L__mod___features_conv2_1_c1x1_a_conv(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_1_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv2_1_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = l__mod___features_conv2_1_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_1_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c3x3_b_bn_running_mean = self.L__mod___features_conv2_1_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c3x3_b_bn_running_var = self.L__mod___features_conv2_1_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_1_c3x3_b_bn_weight = self.L__mod___features_conv2_1_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_1_c3x3_b_bn_bias = self.L__mod___features_conv2_1_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_11 = torch.nn.functional.batch_norm(x_in_1, l__mod___features_conv2_1_c3x3_b_bn_running_mean, l__mod___features_conv2_1_c3x3_b_bn_running_var, l__mod___features_conv2_1_c3x3_b_bn_weight, l__mod___features_conv2_1_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_1 = l__mod___features_conv2_1_c3x3_b_bn_running_mean = l__mod___features_conv2_1_c3x3_b_bn_running_var = l__mod___features_conv2_1_c3x3_b_bn_weight = l__mod___features_conv2_1_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_12 = self.L__mod___features_conv2_1_c3x3_b_bn_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_13 = self.L__mod___features_conv2_1_c3x3_b_bn_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_2 = self.L__mod___features_conv2_1_c3x3_b_conv(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_1_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv2_1_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = l__mod___features_conv2_1_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_1_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_c_bn_running_mean = self.L__mod___features_conv2_1_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_1_c1x1_c_bn_running_var = self.L__mod___features_conv2_1_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_1_c1x1_c_bn_weight = self.L__mod___features_conv2_1_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_1_c1x1_c_bn_bias = self.L__mod___features_conv2_1_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_14 = torch.nn.functional.batch_norm(x_in_2, l__mod___features_conv2_1_c1x1_c_bn_running_mean, l__mod___features_conv2_1_c1x1_c_bn_running_var, l__mod___features_conv2_1_c1x1_c_bn_weight, l__mod___features_conv2_1_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_2 = l__mod___features_conv2_1_c1x1_c_bn_running_mean = l__mod___features_conv2_1_c1x1_c_bn_running_var = l__mod___features_conv2_1_c1x1_c_bn_weight = l__mod___features_conv2_1_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_15 = self.L__mod___features_conv2_1_c1x1_c_bn_drop(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_16 = self.L__mod___features_conv2_1_c1x1_c_bn_act(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_3 = self.L__mod___features_conv2_1_c1x1_c_conv(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1 = x_in_3[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2 = x_in_3[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  x_in_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_1 = x_s1 + out1;  x_s1 = out1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_1 = torch.cat([x_s2, out2], dim = 1);  x_s2 = out2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_4 = torch.cat((x_s1_1, x_s2_1), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_2_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv2_2_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = l__mod___features_conv2_2_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_2_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c1x1_a_bn_running_mean = self.L__mod___features_conv2_2_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c1x1_a_bn_running_var = self.L__mod___features_conv2_2_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_2_c1x1_a_bn_weight = self.L__mod___features_conv2_2_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_2_c1x1_a_bn_bias = self.L__mod___features_conv2_2_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_in_4, l__mod___features_conv2_2_c1x1_a_bn_running_mean, l__mod___features_conv2_2_c1x1_a_bn_running_var, l__mod___features_conv2_2_c1x1_a_bn_weight, l__mod___features_conv2_2_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_4 = l__mod___features_conv2_2_c1x1_a_bn_running_mean = l__mod___features_conv2_2_c1x1_a_bn_running_var = l__mod___features_conv2_2_c1x1_a_bn_weight = l__mod___features_conv2_2_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.L__mod___features_conv2_2_c1x1_a_bn_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_19 = self.L__mod___features_conv2_2_c1x1_a_bn_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_5 = self.L__mod___features_conv2_2_c1x1_a_conv(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_2_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv2_2_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = l__mod___features_conv2_2_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_2_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c3x3_b_bn_running_mean = self.L__mod___features_conv2_2_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c3x3_b_bn_running_var = self.L__mod___features_conv2_2_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_2_c3x3_b_bn_weight = self.L__mod___features_conv2_2_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_2_c3x3_b_bn_bias = self.L__mod___features_conv2_2_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_in_5, l__mod___features_conv2_2_c3x3_b_bn_running_mean, l__mod___features_conv2_2_c3x3_b_bn_running_var, l__mod___features_conv2_2_c3x3_b_bn_weight, l__mod___features_conv2_2_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_5 = l__mod___features_conv2_2_c3x3_b_bn_running_mean = l__mod___features_conv2_2_c3x3_b_bn_running_var = l__mod___features_conv2_2_c3x3_b_bn_weight = l__mod___features_conv2_2_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.L__mod___features_conv2_2_c3x3_b_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_22 = self.L__mod___features_conv2_2_c3x3_b_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_6 = self.L__mod___features_conv2_2_c3x3_b_conv(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_2_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv2_2_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = l__mod___features_conv2_2_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_2_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c1x1_c_bn_running_mean = self.L__mod___features_conv2_2_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_2_c1x1_c_bn_running_var = self.L__mod___features_conv2_2_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_2_c1x1_c_bn_weight = self.L__mod___features_conv2_2_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_2_c1x1_c_bn_bias = self.L__mod___features_conv2_2_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_23 = torch.nn.functional.batch_norm(x_in_6, l__mod___features_conv2_2_c1x1_c_bn_running_mean, l__mod___features_conv2_2_c1x1_c_bn_running_var, l__mod___features_conv2_2_c1x1_c_bn_weight, l__mod___features_conv2_2_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_6 = l__mod___features_conv2_2_c1x1_c_bn_running_mean = l__mod___features_conv2_2_c1x1_c_bn_running_var = l__mod___features_conv2_2_c1x1_c_bn_weight = l__mod___features_conv2_2_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_24 = self.L__mod___features_conv2_2_c1x1_c_bn_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_25 = self.L__mod___features_conv2_2_c1x1_c_bn_act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_7 = self.L__mod___features_conv2_2_c1x1_c_conv(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_1 = x_in_7[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_1 = x_in_7[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  x_in_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_2 = x_s1_1 + out1_1;  x_s1_1 = out1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_2 = torch.cat([x_s2_1, out2_1], dim = 1);  x_s2_1 = out2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_8 = torch.cat((x_s1_2, x_s2_2), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_3_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv2_3_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = l__mod___features_conv2_3_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_3_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c1x1_a_bn_running_mean = self.L__mod___features_conv2_3_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c1x1_a_bn_running_var = self.L__mod___features_conv2_3_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_3_c1x1_a_bn_weight = self.L__mod___features_conv2_3_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_3_c1x1_a_bn_bias = self.L__mod___features_conv2_3_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_in_8, l__mod___features_conv2_3_c1x1_a_bn_running_mean, l__mod___features_conv2_3_c1x1_a_bn_running_var, l__mod___features_conv2_3_c1x1_a_bn_weight, l__mod___features_conv2_3_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_8 = l__mod___features_conv2_3_c1x1_a_bn_running_mean = l__mod___features_conv2_3_c1x1_a_bn_running_var = l__mod___features_conv2_3_c1x1_a_bn_weight = l__mod___features_conv2_3_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.L__mod___features_conv2_3_c1x1_a_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_28 = self.L__mod___features_conv2_3_c1x1_a_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_9 = self.L__mod___features_conv2_3_c1x1_a_conv(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_3_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv2_3_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = l__mod___features_conv2_3_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_3_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c3x3_b_bn_running_mean = self.L__mod___features_conv2_3_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c3x3_b_bn_running_var = self.L__mod___features_conv2_3_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_3_c3x3_b_bn_weight = self.L__mod___features_conv2_3_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_3_c3x3_b_bn_bias = self.L__mod___features_conv2_3_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_29 = torch.nn.functional.batch_norm(x_in_9, l__mod___features_conv2_3_c3x3_b_bn_running_mean, l__mod___features_conv2_3_c3x3_b_bn_running_var, l__mod___features_conv2_3_c3x3_b_bn_weight, l__mod___features_conv2_3_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_9 = l__mod___features_conv2_3_c3x3_b_bn_running_mean = l__mod___features_conv2_3_c3x3_b_bn_running_var = l__mod___features_conv2_3_c3x3_b_bn_weight = l__mod___features_conv2_3_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_30 = self.L__mod___features_conv2_3_c3x3_b_bn_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_31 = self.L__mod___features_conv2_3_c3x3_b_bn_act(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_10 = self.L__mod___features_conv2_3_c3x3_b_conv(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_3_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv2_3_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = l__mod___features_conv2_3_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_3_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c1x1_c_bn_running_mean = self.L__mod___features_conv2_3_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_3_c1x1_c_bn_running_var = self.L__mod___features_conv2_3_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_3_c1x1_c_bn_weight = self.L__mod___features_conv2_3_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_3_c1x1_c_bn_bias = self.L__mod___features_conv2_3_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_32 = torch.nn.functional.batch_norm(x_in_10, l__mod___features_conv2_3_c1x1_c_bn_running_mean, l__mod___features_conv2_3_c1x1_c_bn_running_var, l__mod___features_conv2_3_c1x1_c_bn_weight, l__mod___features_conv2_3_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_10 = l__mod___features_conv2_3_c1x1_c_bn_running_mean = l__mod___features_conv2_3_c1x1_c_bn_running_var = l__mod___features_conv2_3_c1x1_c_bn_weight = l__mod___features_conv2_3_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_33 = self.L__mod___features_conv2_3_c1x1_c_bn_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_34 = self.L__mod___features_conv2_3_c1x1_c_bn_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_11 = self.L__mod___features_conv2_3_c1x1_c_conv(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_2 = x_in_11[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_2 = x_in_11[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  x_in_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_3 = x_s1_2 + out1_2;  x_s1_2 = out1_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_3 = torch.cat([x_s2_2, out2_2], dim = 1);  x_s2_2 = out2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_12 = torch.cat((x_s1_3, x_s2_3), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_4_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv2_4_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = l__mod___features_conv2_4_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_4_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c1x1_a_bn_running_mean = self.L__mod___features_conv2_4_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c1x1_a_bn_running_var = self.L__mod___features_conv2_4_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_4_c1x1_a_bn_weight = self.L__mod___features_conv2_4_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_4_c1x1_a_bn_bias = self.L__mod___features_conv2_4_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_35 = torch.nn.functional.batch_norm(x_in_12, l__mod___features_conv2_4_c1x1_a_bn_running_mean, l__mod___features_conv2_4_c1x1_a_bn_running_var, l__mod___features_conv2_4_c1x1_a_bn_weight, l__mod___features_conv2_4_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_12 = l__mod___features_conv2_4_c1x1_a_bn_running_mean = l__mod___features_conv2_4_c1x1_a_bn_running_var = l__mod___features_conv2_4_c1x1_a_bn_weight = l__mod___features_conv2_4_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_36 = self.L__mod___features_conv2_4_c1x1_a_bn_drop(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_37 = self.L__mod___features_conv2_4_c1x1_a_bn_act(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_13 = self.L__mod___features_conv2_4_c1x1_a_conv(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_4_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv2_4_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = l__mod___features_conv2_4_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_4_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c3x3_b_bn_running_mean = self.L__mod___features_conv2_4_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c3x3_b_bn_running_var = self.L__mod___features_conv2_4_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_4_c3x3_b_bn_weight = self.L__mod___features_conv2_4_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_4_c3x3_b_bn_bias = self.L__mod___features_conv2_4_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_38 = torch.nn.functional.batch_norm(x_in_13, l__mod___features_conv2_4_c3x3_b_bn_running_mean, l__mod___features_conv2_4_c3x3_b_bn_running_var, l__mod___features_conv2_4_c3x3_b_bn_weight, l__mod___features_conv2_4_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_13 = l__mod___features_conv2_4_c3x3_b_bn_running_mean = l__mod___features_conv2_4_c3x3_b_bn_running_var = l__mod___features_conv2_4_c3x3_b_bn_weight = l__mod___features_conv2_4_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_39 = self.L__mod___features_conv2_4_c3x3_b_bn_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_40 = self.L__mod___features_conv2_4_c3x3_b_bn_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_14 = self.L__mod___features_conv2_4_c3x3_b_conv(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv2_4_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv2_4_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = l__mod___features_conv2_4_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv2_4_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c1x1_c_bn_running_mean = self.L__mod___features_conv2_4_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv2_4_c1x1_c_bn_running_var = self.L__mod___features_conv2_4_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv2_4_c1x1_c_bn_weight = self.L__mod___features_conv2_4_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv2_4_c1x1_c_bn_bias = self.L__mod___features_conv2_4_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_41 = torch.nn.functional.batch_norm(x_in_14, l__mod___features_conv2_4_c1x1_c_bn_running_mean, l__mod___features_conv2_4_c1x1_c_bn_running_var, l__mod___features_conv2_4_c1x1_c_bn_weight, l__mod___features_conv2_4_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_14 = l__mod___features_conv2_4_c1x1_c_bn_running_mean = l__mod___features_conv2_4_c1x1_c_bn_running_var = l__mod___features_conv2_4_c1x1_c_bn_weight = l__mod___features_conv2_4_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_42 = self.L__mod___features_conv2_4_c1x1_c_bn_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_43 = self.L__mod___features_conv2_4_c1x1_c_bn_act(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_15 = self.L__mod___features_conv2_4_c1x1_c_conv(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_3 = x_in_15[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_3 = x_in_15[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  x_in_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    resid_3 = x_s1_3 + out1_3;  x_s1_3 = out1_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    dense_3 = torch.cat([x_s2_3, out2_3], dim = 1);  x_s2_3 = out2_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_16 = torch.cat((resid_3, dense_3), dim = 1);  resid_3 = dense_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_1_c1x1_w_s2_bn_num_batches_tracked = self.L__mod___features_conv3_1_c1x1_w_s2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = l__mod___features_conv3_1_c1x1_w_s2_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_1_c1x1_w_s2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_w_s2_bn_running_mean = self.L__mod___features_conv3_1_c1x1_w_s2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_w_s2_bn_running_var = self.L__mod___features_conv3_1_c1x1_w_s2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_1_c1x1_w_s2_bn_weight = self.L__mod___features_conv3_1_c1x1_w_s2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_1_c1x1_w_s2_bn_bias = self.L__mod___features_conv3_1_c1x1_w_s2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_44 = torch.nn.functional.batch_norm(x_in_16, l__mod___features_conv3_1_c1x1_w_s2_bn_running_mean, l__mod___features_conv3_1_c1x1_w_s2_bn_running_var, l__mod___features_conv3_1_c1x1_w_s2_bn_weight, l__mod___features_conv3_1_c1x1_w_s2_bn_bias, True, 0.1, 0.001);  l__mod___features_conv3_1_c1x1_w_s2_bn_running_mean = l__mod___features_conv3_1_c1x1_w_s2_bn_running_var = l__mod___features_conv3_1_c1x1_w_s2_bn_weight = l__mod___features_conv3_1_c1x1_w_s2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_45 = self.L__mod___features_conv3_1_c1x1_w_s2_bn_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_46 = self.L__mod___features_conv3_1_c1x1_w_s2_bn_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_s_1 = self.L__mod___features_conv3_1_c1x1_w_s2_conv(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    x_s1_4 = x_s_1[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    x_s2_4 = x_s_1[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_s_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_1_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_1_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = l__mod___features_conv3_1_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_1_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_a_bn_running_mean = self.L__mod___features_conv3_1_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_a_bn_running_var = self.L__mod___features_conv3_1_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_1_c1x1_a_bn_weight = self.L__mod___features_conv3_1_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_1_c1x1_a_bn_bias = self.L__mod___features_conv3_1_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_47 = torch.nn.functional.batch_norm(x_in_16, l__mod___features_conv3_1_c1x1_a_bn_running_mean, l__mod___features_conv3_1_c1x1_a_bn_running_var, l__mod___features_conv3_1_c1x1_a_bn_weight, l__mod___features_conv3_1_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_16 = l__mod___features_conv3_1_c1x1_a_bn_running_mean = l__mod___features_conv3_1_c1x1_a_bn_running_var = l__mod___features_conv3_1_c1x1_a_bn_weight = l__mod___features_conv3_1_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_48 = self.L__mod___features_conv3_1_c1x1_a_bn_drop(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.L__mod___features_conv3_1_c1x1_a_bn_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_17 = self.L__mod___features_conv3_1_c1x1_a_conv(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_1_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_1_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = l__mod___features_conv3_1_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_1_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c3x3_b_bn_running_mean = self.L__mod___features_conv3_1_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c3x3_b_bn_running_var = self.L__mod___features_conv3_1_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_1_c3x3_b_bn_weight = self.L__mod___features_conv3_1_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_1_c3x3_b_bn_bias = self.L__mod___features_conv3_1_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_50 = torch.nn.functional.batch_norm(x_in_17, l__mod___features_conv3_1_c3x3_b_bn_running_mean, l__mod___features_conv3_1_c3x3_b_bn_running_var, l__mod___features_conv3_1_c3x3_b_bn_weight, l__mod___features_conv3_1_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_17 = l__mod___features_conv3_1_c3x3_b_bn_running_mean = l__mod___features_conv3_1_c3x3_b_bn_running_var = l__mod___features_conv3_1_c3x3_b_bn_weight = l__mod___features_conv3_1_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_51 = self.L__mod___features_conv3_1_c3x3_b_bn_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_52 = self.L__mod___features_conv3_1_c3x3_b_bn_act(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_18 = self.L__mod___features_conv3_1_c3x3_b_conv(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_1_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_1_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = l__mod___features_conv3_1_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_1_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_c_bn_running_mean = self.L__mod___features_conv3_1_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_1_c1x1_c_bn_running_var = self.L__mod___features_conv3_1_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_1_c1x1_c_bn_weight = self.L__mod___features_conv3_1_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_1_c1x1_c_bn_bias = self.L__mod___features_conv3_1_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_53 = torch.nn.functional.batch_norm(x_in_18, l__mod___features_conv3_1_c1x1_c_bn_running_mean, l__mod___features_conv3_1_c1x1_c_bn_running_var, l__mod___features_conv3_1_c1x1_c_bn_weight, l__mod___features_conv3_1_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_18 = l__mod___features_conv3_1_c1x1_c_bn_running_mean = l__mod___features_conv3_1_c1x1_c_bn_running_var = l__mod___features_conv3_1_c1x1_c_bn_weight = l__mod___features_conv3_1_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_54 = self.L__mod___features_conv3_1_c1x1_c_bn_drop(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.L__mod___features_conv3_1_c1x1_c_bn_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_19 = self.L__mod___features_conv3_1_c1x1_c_conv(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_4 = x_in_19[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_4 = x_in_19[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_5 = x_s1_4 + out1_4;  x_s1_4 = out1_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_5 = torch.cat([x_s2_4, out2_4], dim = 1);  x_s2_4 = out2_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_20 = torch.cat((x_s1_5, x_s2_5), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_2_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_2_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = l__mod___features_conv3_2_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_2_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c1x1_a_bn_running_mean = self.L__mod___features_conv3_2_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c1x1_a_bn_running_var = self.L__mod___features_conv3_2_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_2_c1x1_a_bn_weight = self.L__mod___features_conv3_2_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_2_c1x1_a_bn_bias = self.L__mod___features_conv3_2_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_56 = torch.nn.functional.batch_norm(x_in_20, l__mod___features_conv3_2_c1x1_a_bn_running_mean, l__mod___features_conv3_2_c1x1_a_bn_running_var, l__mod___features_conv3_2_c1x1_a_bn_weight, l__mod___features_conv3_2_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_20 = l__mod___features_conv3_2_c1x1_a_bn_running_mean = l__mod___features_conv3_2_c1x1_a_bn_running_var = l__mod___features_conv3_2_c1x1_a_bn_weight = l__mod___features_conv3_2_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_57 = self.L__mod___features_conv3_2_c1x1_a_bn_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_58 = self.L__mod___features_conv3_2_c1x1_a_bn_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_21 = self.L__mod___features_conv3_2_c1x1_a_conv(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_2_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_2_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = l__mod___features_conv3_2_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_2_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c3x3_b_bn_running_mean = self.L__mod___features_conv3_2_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c3x3_b_bn_running_var = self.L__mod___features_conv3_2_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_2_c3x3_b_bn_weight = self.L__mod___features_conv3_2_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_2_c3x3_b_bn_bias = self.L__mod___features_conv3_2_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_59 = torch.nn.functional.batch_norm(x_in_21, l__mod___features_conv3_2_c3x3_b_bn_running_mean, l__mod___features_conv3_2_c3x3_b_bn_running_var, l__mod___features_conv3_2_c3x3_b_bn_weight, l__mod___features_conv3_2_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_21 = l__mod___features_conv3_2_c3x3_b_bn_running_mean = l__mod___features_conv3_2_c3x3_b_bn_running_var = l__mod___features_conv3_2_c3x3_b_bn_weight = l__mod___features_conv3_2_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_60 = self.L__mod___features_conv3_2_c3x3_b_bn_drop(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_61 = self.L__mod___features_conv3_2_c3x3_b_bn_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_22 = self.L__mod___features_conv3_2_c3x3_b_conv(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_2_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_2_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = l__mod___features_conv3_2_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_2_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c1x1_c_bn_running_mean = self.L__mod___features_conv3_2_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_2_c1x1_c_bn_running_var = self.L__mod___features_conv3_2_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_2_c1x1_c_bn_weight = self.L__mod___features_conv3_2_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_2_c1x1_c_bn_bias = self.L__mod___features_conv3_2_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_62 = torch.nn.functional.batch_norm(x_in_22, l__mod___features_conv3_2_c1x1_c_bn_running_mean, l__mod___features_conv3_2_c1x1_c_bn_running_var, l__mod___features_conv3_2_c1x1_c_bn_weight, l__mod___features_conv3_2_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_22 = l__mod___features_conv3_2_c1x1_c_bn_running_mean = l__mod___features_conv3_2_c1x1_c_bn_running_var = l__mod___features_conv3_2_c1x1_c_bn_weight = l__mod___features_conv3_2_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_63 = self.L__mod___features_conv3_2_c1x1_c_bn_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_64 = self.L__mod___features_conv3_2_c1x1_c_bn_act(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_23 = self.L__mod___features_conv3_2_c1x1_c_conv(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_5 = x_in_23[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_5 = x_in_23[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_6 = x_s1_5 + out1_5;  x_s1_5 = out1_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_6 = torch.cat([x_s2_5, out2_5], dim = 1);  x_s2_5 = out2_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_24 = torch.cat((x_s1_6, x_s2_6), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_3_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_3_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = l__mod___features_conv3_3_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_3_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c1x1_a_bn_running_mean = self.L__mod___features_conv3_3_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c1x1_a_bn_running_var = self.L__mod___features_conv3_3_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_3_c1x1_a_bn_weight = self.L__mod___features_conv3_3_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_3_c1x1_a_bn_bias = self.L__mod___features_conv3_3_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_65 = torch.nn.functional.batch_norm(x_in_24, l__mod___features_conv3_3_c1x1_a_bn_running_mean, l__mod___features_conv3_3_c1x1_a_bn_running_var, l__mod___features_conv3_3_c1x1_a_bn_weight, l__mod___features_conv3_3_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_24 = l__mod___features_conv3_3_c1x1_a_bn_running_mean = l__mod___features_conv3_3_c1x1_a_bn_running_var = l__mod___features_conv3_3_c1x1_a_bn_weight = l__mod___features_conv3_3_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_66 = self.L__mod___features_conv3_3_c1x1_a_bn_drop(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_67 = self.L__mod___features_conv3_3_c1x1_a_bn_act(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_25 = self.L__mod___features_conv3_3_c1x1_a_conv(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_3_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_3_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = l__mod___features_conv3_3_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_3_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c3x3_b_bn_running_mean = self.L__mod___features_conv3_3_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c3x3_b_bn_running_var = self.L__mod___features_conv3_3_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_3_c3x3_b_bn_weight = self.L__mod___features_conv3_3_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_3_c3x3_b_bn_bias = self.L__mod___features_conv3_3_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_68 = torch.nn.functional.batch_norm(x_in_25, l__mod___features_conv3_3_c3x3_b_bn_running_mean, l__mod___features_conv3_3_c3x3_b_bn_running_var, l__mod___features_conv3_3_c3x3_b_bn_weight, l__mod___features_conv3_3_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_25 = l__mod___features_conv3_3_c3x3_b_bn_running_mean = l__mod___features_conv3_3_c3x3_b_bn_running_var = l__mod___features_conv3_3_c3x3_b_bn_weight = l__mod___features_conv3_3_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_69 = self.L__mod___features_conv3_3_c3x3_b_bn_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_70 = self.L__mod___features_conv3_3_c3x3_b_bn_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_26 = self.L__mod___features_conv3_3_c3x3_b_conv(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_3_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_3_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = l__mod___features_conv3_3_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_3_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c1x1_c_bn_running_mean = self.L__mod___features_conv3_3_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_3_c1x1_c_bn_running_var = self.L__mod___features_conv3_3_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_3_c1x1_c_bn_weight = self.L__mod___features_conv3_3_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_3_c1x1_c_bn_bias = self.L__mod___features_conv3_3_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_71 = torch.nn.functional.batch_norm(x_in_26, l__mod___features_conv3_3_c1x1_c_bn_running_mean, l__mod___features_conv3_3_c1x1_c_bn_running_var, l__mod___features_conv3_3_c1x1_c_bn_weight, l__mod___features_conv3_3_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_26 = l__mod___features_conv3_3_c1x1_c_bn_running_mean = l__mod___features_conv3_3_c1x1_c_bn_running_var = l__mod___features_conv3_3_c1x1_c_bn_weight = l__mod___features_conv3_3_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_72 = self.L__mod___features_conv3_3_c1x1_c_bn_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_73 = self.L__mod___features_conv3_3_c1x1_c_bn_act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_27 = self.L__mod___features_conv3_3_c1x1_c_conv(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_6 = x_in_27[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_6 = x_in_27[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_7 = x_s1_6 + out1_6;  x_s1_6 = out1_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_7 = torch.cat([x_s2_6, out2_6], dim = 1);  x_s2_6 = out2_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_28 = torch.cat((x_s1_7, x_s2_7), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_4_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_4_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = l__mod___features_conv3_4_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_4_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c1x1_a_bn_running_mean = self.L__mod___features_conv3_4_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c1x1_a_bn_running_var = self.L__mod___features_conv3_4_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_4_c1x1_a_bn_weight = self.L__mod___features_conv3_4_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_4_c1x1_a_bn_bias = self.L__mod___features_conv3_4_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_74 = torch.nn.functional.batch_norm(x_in_28, l__mod___features_conv3_4_c1x1_a_bn_running_mean, l__mod___features_conv3_4_c1x1_a_bn_running_var, l__mod___features_conv3_4_c1x1_a_bn_weight, l__mod___features_conv3_4_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_28 = l__mod___features_conv3_4_c1x1_a_bn_running_mean = l__mod___features_conv3_4_c1x1_a_bn_running_var = l__mod___features_conv3_4_c1x1_a_bn_weight = l__mod___features_conv3_4_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.L__mod___features_conv3_4_c1x1_a_bn_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_76 = self.L__mod___features_conv3_4_c1x1_a_bn_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_29 = self.L__mod___features_conv3_4_c1x1_a_conv(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_4_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_4_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = l__mod___features_conv3_4_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_4_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c3x3_b_bn_running_mean = self.L__mod___features_conv3_4_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c3x3_b_bn_running_var = self.L__mod___features_conv3_4_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_4_c3x3_b_bn_weight = self.L__mod___features_conv3_4_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_4_c3x3_b_bn_bias = self.L__mod___features_conv3_4_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_77 = torch.nn.functional.batch_norm(x_in_29, l__mod___features_conv3_4_c3x3_b_bn_running_mean, l__mod___features_conv3_4_c3x3_b_bn_running_var, l__mod___features_conv3_4_c3x3_b_bn_weight, l__mod___features_conv3_4_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_29 = l__mod___features_conv3_4_c3x3_b_bn_running_mean = l__mod___features_conv3_4_c3x3_b_bn_running_var = l__mod___features_conv3_4_c3x3_b_bn_weight = l__mod___features_conv3_4_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_78 = self.L__mod___features_conv3_4_c3x3_b_bn_drop(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_79 = self.L__mod___features_conv3_4_c3x3_b_bn_act(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_30 = self.L__mod___features_conv3_4_c3x3_b_conv(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_4_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_4_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = l__mod___features_conv3_4_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_4_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c1x1_c_bn_running_mean = self.L__mod___features_conv3_4_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_4_c1x1_c_bn_running_var = self.L__mod___features_conv3_4_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_4_c1x1_c_bn_weight = self.L__mod___features_conv3_4_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_4_c1x1_c_bn_bias = self.L__mod___features_conv3_4_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_80 = torch.nn.functional.batch_norm(x_in_30, l__mod___features_conv3_4_c1x1_c_bn_running_mean, l__mod___features_conv3_4_c1x1_c_bn_running_var, l__mod___features_conv3_4_c1x1_c_bn_weight, l__mod___features_conv3_4_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_30 = l__mod___features_conv3_4_c1x1_c_bn_running_mean = l__mod___features_conv3_4_c1x1_c_bn_running_var = l__mod___features_conv3_4_c1x1_c_bn_weight = l__mod___features_conv3_4_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_81 = self.L__mod___features_conv3_4_c1x1_c_bn_drop(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_82 = self.L__mod___features_conv3_4_c1x1_c_bn_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_31 = self.L__mod___features_conv3_4_c1x1_c_conv(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_7 = x_in_31[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_7 = x_in_31[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_8 = x_s1_7 + out1_7;  x_s1_7 = out1_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_8 = torch.cat([x_s2_7, out2_7], dim = 1);  x_s2_7 = out2_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_32 = torch.cat((x_s1_8, x_s2_8), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_5_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_5_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = l__mod___features_conv3_5_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_5_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c1x1_a_bn_running_mean = self.L__mod___features_conv3_5_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c1x1_a_bn_running_var = self.L__mod___features_conv3_5_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_5_c1x1_a_bn_weight = self.L__mod___features_conv3_5_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_5_c1x1_a_bn_bias = self.L__mod___features_conv3_5_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_83 = torch.nn.functional.batch_norm(x_in_32, l__mod___features_conv3_5_c1x1_a_bn_running_mean, l__mod___features_conv3_5_c1x1_a_bn_running_var, l__mod___features_conv3_5_c1x1_a_bn_weight, l__mod___features_conv3_5_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_32 = l__mod___features_conv3_5_c1x1_a_bn_running_mean = l__mod___features_conv3_5_c1x1_a_bn_running_var = l__mod___features_conv3_5_c1x1_a_bn_weight = l__mod___features_conv3_5_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_84 = self.L__mod___features_conv3_5_c1x1_a_bn_drop(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_85 = self.L__mod___features_conv3_5_c1x1_a_bn_act(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_33 = self.L__mod___features_conv3_5_c1x1_a_conv(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_5_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_5_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = l__mod___features_conv3_5_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_5_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c3x3_b_bn_running_mean = self.L__mod___features_conv3_5_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c3x3_b_bn_running_var = self.L__mod___features_conv3_5_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_5_c3x3_b_bn_weight = self.L__mod___features_conv3_5_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_5_c3x3_b_bn_bias = self.L__mod___features_conv3_5_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_86 = torch.nn.functional.batch_norm(x_in_33, l__mod___features_conv3_5_c3x3_b_bn_running_mean, l__mod___features_conv3_5_c3x3_b_bn_running_var, l__mod___features_conv3_5_c3x3_b_bn_weight, l__mod___features_conv3_5_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_33 = l__mod___features_conv3_5_c3x3_b_bn_running_mean = l__mod___features_conv3_5_c3x3_b_bn_running_var = l__mod___features_conv3_5_c3x3_b_bn_weight = l__mod___features_conv3_5_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_87 = self.L__mod___features_conv3_5_c3x3_b_bn_drop(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_88 = self.L__mod___features_conv3_5_c3x3_b_bn_act(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_34 = self.L__mod___features_conv3_5_c3x3_b_conv(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_5_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_5_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = l__mod___features_conv3_5_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_5_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c1x1_c_bn_running_mean = self.L__mod___features_conv3_5_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_5_c1x1_c_bn_running_var = self.L__mod___features_conv3_5_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_5_c1x1_c_bn_weight = self.L__mod___features_conv3_5_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_5_c1x1_c_bn_bias = self.L__mod___features_conv3_5_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_in_34, l__mod___features_conv3_5_c1x1_c_bn_running_mean, l__mod___features_conv3_5_c1x1_c_bn_running_var, l__mod___features_conv3_5_c1x1_c_bn_weight, l__mod___features_conv3_5_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_34 = l__mod___features_conv3_5_c1x1_c_bn_running_mean = l__mod___features_conv3_5_c1x1_c_bn_running_var = l__mod___features_conv3_5_c1x1_c_bn_weight = l__mod___features_conv3_5_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.L__mod___features_conv3_5_c1x1_c_bn_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_91 = self.L__mod___features_conv3_5_c1x1_c_bn_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_35 = self.L__mod___features_conv3_5_c1x1_c_conv(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_8 = x_in_35[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_8 = x_in_35[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_9 = x_s1_8 + out1_8;  x_s1_8 = out1_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_9 = torch.cat([x_s2_8, out2_8], dim = 1);  x_s2_8 = out2_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_36 = torch.cat((x_s1_9, x_s2_9), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_6_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_6_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = l__mod___features_conv3_6_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_6_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c1x1_a_bn_running_mean = self.L__mod___features_conv3_6_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c1x1_a_bn_running_var = self.L__mod___features_conv3_6_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_6_c1x1_a_bn_weight = self.L__mod___features_conv3_6_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_6_c1x1_a_bn_bias = self.L__mod___features_conv3_6_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_92 = torch.nn.functional.batch_norm(x_in_36, l__mod___features_conv3_6_c1x1_a_bn_running_mean, l__mod___features_conv3_6_c1x1_a_bn_running_var, l__mod___features_conv3_6_c1x1_a_bn_weight, l__mod___features_conv3_6_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_36 = l__mod___features_conv3_6_c1x1_a_bn_running_mean = l__mod___features_conv3_6_c1x1_a_bn_running_var = l__mod___features_conv3_6_c1x1_a_bn_weight = l__mod___features_conv3_6_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_93 = self.L__mod___features_conv3_6_c1x1_a_bn_drop(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_94 = self.L__mod___features_conv3_6_c1x1_a_bn_act(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_37 = self.L__mod___features_conv3_6_c1x1_a_conv(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_6_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_6_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = l__mod___features_conv3_6_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_6_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c3x3_b_bn_running_mean = self.L__mod___features_conv3_6_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c3x3_b_bn_running_var = self.L__mod___features_conv3_6_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_6_c3x3_b_bn_weight = self.L__mod___features_conv3_6_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_6_c3x3_b_bn_bias = self.L__mod___features_conv3_6_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_95 = torch.nn.functional.batch_norm(x_in_37, l__mod___features_conv3_6_c3x3_b_bn_running_mean, l__mod___features_conv3_6_c3x3_b_bn_running_var, l__mod___features_conv3_6_c3x3_b_bn_weight, l__mod___features_conv3_6_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_37 = l__mod___features_conv3_6_c3x3_b_bn_running_mean = l__mod___features_conv3_6_c3x3_b_bn_running_var = l__mod___features_conv3_6_c3x3_b_bn_weight = l__mod___features_conv3_6_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_96 = self.L__mod___features_conv3_6_c3x3_b_bn_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_97 = self.L__mod___features_conv3_6_c3x3_b_bn_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_38 = self.L__mod___features_conv3_6_c3x3_b_conv(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_6_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_6_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__32 = l__mod___features_conv3_6_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_6_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c1x1_c_bn_running_mean = self.L__mod___features_conv3_6_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_6_c1x1_c_bn_running_var = self.L__mod___features_conv3_6_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_6_c1x1_c_bn_weight = self.L__mod___features_conv3_6_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_6_c1x1_c_bn_bias = self.L__mod___features_conv3_6_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_98 = torch.nn.functional.batch_norm(x_in_38, l__mod___features_conv3_6_c1x1_c_bn_running_mean, l__mod___features_conv3_6_c1x1_c_bn_running_var, l__mod___features_conv3_6_c1x1_c_bn_weight, l__mod___features_conv3_6_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_38 = l__mod___features_conv3_6_c1x1_c_bn_running_mean = l__mod___features_conv3_6_c1x1_c_bn_running_var = l__mod___features_conv3_6_c1x1_c_bn_weight = l__mod___features_conv3_6_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_99 = self.L__mod___features_conv3_6_c1x1_c_bn_drop(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.L__mod___features_conv3_6_c1x1_c_bn_act(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_39 = self.L__mod___features_conv3_6_c1x1_c_conv(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_9 = x_in_39[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_9 = x_in_39[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_10 = x_s1_9 + out1_9;  x_s1_9 = out1_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_10 = torch.cat([x_s2_9, out2_9], dim = 1);  x_s2_9 = out2_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_40 = torch.cat((x_s1_10, x_s2_10), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_7_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_7_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__33 = l__mod___features_conv3_7_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_7_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c1x1_a_bn_running_mean = self.L__mod___features_conv3_7_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c1x1_a_bn_running_var = self.L__mod___features_conv3_7_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_7_c1x1_a_bn_weight = self.L__mod___features_conv3_7_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_7_c1x1_a_bn_bias = self.L__mod___features_conv3_7_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_101 = torch.nn.functional.batch_norm(x_in_40, l__mod___features_conv3_7_c1x1_a_bn_running_mean, l__mod___features_conv3_7_c1x1_a_bn_running_var, l__mod___features_conv3_7_c1x1_a_bn_weight, l__mod___features_conv3_7_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_40 = l__mod___features_conv3_7_c1x1_a_bn_running_mean = l__mod___features_conv3_7_c1x1_a_bn_running_var = l__mod___features_conv3_7_c1x1_a_bn_weight = l__mod___features_conv3_7_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_102 = self.L__mod___features_conv3_7_c1x1_a_bn_drop(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_103 = self.L__mod___features_conv3_7_c1x1_a_bn_act(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_41 = self.L__mod___features_conv3_7_c1x1_a_conv(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_7_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_7_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__34 = l__mod___features_conv3_7_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_7_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c3x3_b_bn_running_mean = self.L__mod___features_conv3_7_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c3x3_b_bn_running_var = self.L__mod___features_conv3_7_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_7_c3x3_b_bn_weight = self.L__mod___features_conv3_7_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_7_c3x3_b_bn_bias = self.L__mod___features_conv3_7_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_104 = torch.nn.functional.batch_norm(x_in_41, l__mod___features_conv3_7_c3x3_b_bn_running_mean, l__mod___features_conv3_7_c3x3_b_bn_running_var, l__mod___features_conv3_7_c3x3_b_bn_weight, l__mod___features_conv3_7_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_41 = l__mod___features_conv3_7_c3x3_b_bn_running_mean = l__mod___features_conv3_7_c3x3_b_bn_running_var = l__mod___features_conv3_7_c3x3_b_bn_weight = l__mod___features_conv3_7_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_105 = self.L__mod___features_conv3_7_c3x3_b_bn_drop(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_106 = self.L__mod___features_conv3_7_c3x3_b_bn_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_42 = self.L__mod___features_conv3_7_c3x3_b_conv(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_7_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_7_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__35 = l__mod___features_conv3_7_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_7_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c1x1_c_bn_running_mean = self.L__mod___features_conv3_7_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_7_c1x1_c_bn_running_var = self.L__mod___features_conv3_7_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_7_c1x1_c_bn_weight = self.L__mod___features_conv3_7_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_7_c1x1_c_bn_bias = self.L__mod___features_conv3_7_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_107 = torch.nn.functional.batch_norm(x_in_42, l__mod___features_conv3_7_c1x1_c_bn_running_mean, l__mod___features_conv3_7_c1x1_c_bn_running_var, l__mod___features_conv3_7_c1x1_c_bn_weight, l__mod___features_conv3_7_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_42 = l__mod___features_conv3_7_c1x1_c_bn_running_mean = l__mod___features_conv3_7_c1x1_c_bn_running_var = l__mod___features_conv3_7_c1x1_c_bn_weight = l__mod___features_conv3_7_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_108 = self.L__mod___features_conv3_7_c1x1_c_bn_drop(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_109 = self.L__mod___features_conv3_7_c1x1_c_bn_act(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_43 = self.L__mod___features_conv3_7_c1x1_c_conv(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_10 = x_in_43[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_10 = x_in_43[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_11 = x_s1_10 + out1_10;  x_s1_10 = out1_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_11 = torch.cat([x_s2_10, out2_10], dim = 1);  x_s2_10 = out2_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_44 = torch.cat((x_s1_11, x_s2_11), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_8_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv3_8_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__36 = l__mod___features_conv3_8_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_8_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c1x1_a_bn_running_mean = self.L__mod___features_conv3_8_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c1x1_a_bn_running_var = self.L__mod___features_conv3_8_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_8_c1x1_a_bn_weight = self.L__mod___features_conv3_8_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_8_c1x1_a_bn_bias = self.L__mod___features_conv3_8_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_110 = torch.nn.functional.batch_norm(x_in_44, l__mod___features_conv3_8_c1x1_a_bn_running_mean, l__mod___features_conv3_8_c1x1_a_bn_running_var, l__mod___features_conv3_8_c1x1_a_bn_weight, l__mod___features_conv3_8_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_44 = l__mod___features_conv3_8_c1x1_a_bn_running_mean = l__mod___features_conv3_8_c1x1_a_bn_running_var = l__mod___features_conv3_8_c1x1_a_bn_weight = l__mod___features_conv3_8_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_111 = self.L__mod___features_conv3_8_c1x1_a_bn_drop(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_112 = self.L__mod___features_conv3_8_c1x1_a_bn_act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_45 = self.L__mod___features_conv3_8_c1x1_a_conv(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_8_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv3_8_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__37 = l__mod___features_conv3_8_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_8_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c3x3_b_bn_running_mean = self.L__mod___features_conv3_8_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c3x3_b_bn_running_var = self.L__mod___features_conv3_8_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_8_c3x3_b_bn_weight = self.L__mod___features_conv3_8_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_8_c3x3_b_bn_bias = self.L__mod___features_conv3_8_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_113 = torch.nn.functional.batch_norm(x_in_45, l__mod___features_conv3_8_c3x3_b_bn_running_mean, l__mod___features_conv3_8_c3x3_b_bn_running_var, l__mod___features_conv3_8_c3x3_b_bn_weight, l__mod___features_conv3_8_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_45 = l__mod___features_conv3_8_c3x3_b_bn_running_mean = l__mod___features_conv3_8_c3x3_b_bn_running_var = l__mod___features_conv3_8_c3x3_b_bn_weight = l__mod___features_conv3_8_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_114 = self.L__mod___features_conv3_8_c3x3_b_bn_drop(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_115 = self.L__mod___features_conv3_8_c3x3_b_bn_act(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_46 = self.L__mod___features_conv3_8_c3x3_b_conv(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv3_8_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv3_8_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__38 = l__mod___features_conv3_8_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv3_8_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c1x1_c_bn_running_mean = self.L__mod___features_conv3_8_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv3_8_c1x1_c_bn_running_var = self.L__mod___features_conv3_8_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv3_8_c1x1_c_bn_weight = self.L__mod___features_conv3_8_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv3_8_c1x1_c_bn_bias = self.L__mod___features_conv3_8_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_116 = torch.nn.functional.batch_norm(x_in_46, l__mod___features_conv3_8_c1x1_c_bn_running_mean, l__mod___features_conv3_8_c1x1_c_bn_running_var, l__mod___features_conv3_8_c1x1_c_bn_weight, l__mod___features_conv3_8_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_46 = l__mod___features_conv3_8_c1x1_c_bn_running_mean = l__mod___features_conv3_8_c1x1_c_bn_running_var = l__mod___features_conv3_8_c1x1_c_bn_weight = l__mod___features_conv3_8_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_117 = self.L__mod___features_conv3_8_c1x1_c_bn_drop(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_118 = self.L__mod___features_conv3_8_c1x1_c_bn_act(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_47 = self.L__mod___features_conv3_8_c1x1_c_conv(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_11 = x_in_47[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_11 = x_in_47[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  x_in_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    resid_11 = x_s1_11 + out1_11;  x_s1_11 = out1_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    dense_11 = torch.cat([x_s2_11, out2_11], dim = 1);  x_s2_11 = out2_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_48 = torch.cat((resid_11, dense_11), dim = 1);  resid_11 = dense_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_1_c1x1_w_s2_bn_num_batches_tracked = self.L__mod___features_conv4_1_c1x1_w_s2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__39 = l__mod___features_conv4_1_c1x1_w_s2_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_1_c1x1_w_s2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_w_s2_bn_running_mean = self.L__mod___features_conv4_1_c1x1_w_s2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_w_s2_bn_running_var = self.L__mod___features_conv4_1_c1x1_w_s2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_1_c1x1_w_s2_bn_weight = self.L__mod___features_conv4_1_c1x1_w_s2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_1_c1x1_w_s2_bn_bias = self.L__mod___features_conv4_1_c1x1_w_s2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_in_48, l__mod___features_conv4_1_c1x1_w_s2_bn_running_mean, l__mod___features_conv4_1_c1x1_w_s2_bn_running_var, l__mod___features_conv4_1_c1x1_w_s2_bn_weight, l__mod___features_conv4_1_c1x1_w_s2_bn_bias, True, 0.1, 0.001);  l__mod___features_conv4_1_c1x1_w_s2_bn_running_mean = l__mod___features_conv4_1_c1x1_w_s2_bn_running_var = l__mod___features_conv4_1_c1x1_w_s2_bn_weight = l__mod___features_conv4_1_c1x1_w_s2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.L__mod___features_conv4_1_c1x1_w_s2_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_121 = self.L__mod___features_conv4_1_c1x1_w_s2_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_s_2 = self.L__mod___features_conv4_1_c1x1_w_s2_conv(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    x_s1_12 = x_s_2[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    x_s2_12 = x_s_2[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_s_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_1_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_1_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__40 = l__mod___features_conv4_1_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_1_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_a_bn_running_mean = self.L__mod___features_conv4_1_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_a_bn_running_var = self.L__mod___features_conv4_1_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_1_c1x1_a_bn_weight = self.L__mod___features_conv4_1_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_1_c1x1_a_bn_bias = self.L__mod___features_conv4_1_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_122 = torch.nn.functional.batch_norm(x_in_48, l__mod___features_conv4_1_c1x1_a_bn_running_mean, l__mod___features_conv4_1_c1x1_a_bn_running_var, l__mod___features_conv4_1_c1x1_a_bn_weight, l__mod___features_conv4_1_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_48 = l__mod___features_conv4_1_c1x1_a_bn_running_mean = l__mod___features_conv4_1_c1x1_a_bn_running_var = l__mod___features_conv4_1_c1x1_a_bn_weight = l__mod___features_conv4_1_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_123 = self.L__mod___features_conv4_1_c1x1_a_bn_drop(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_124 = self.L__mod___features_conv4_1_c1x1_a_bn_act(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_49 = self.L__mod___features_conv4_1_c1x1_a_conv(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_1_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_1_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__41 = l__mod___features_conv4_1_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_1_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c3x3_b_bn_running_mean = self.L__mod___features_conv4_1_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c3x3_b_bn_running_var = self.L__mod___features_conv4_1_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_1_c3x3_b_bn_weight = self.L__mod___features_conv4_1_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_1_c3x3_b_bn_bias = self.L__mod___features_conv4_1_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_125 = torch.nn.functional.batch_norm(x_in_49, l__mod___features_conv4_1_c3x3_b_bn_running_mean, l__mod___features_conv4_1_c3x3_b_bn_running_var, l__mod___features_conv4_1_c3x3_b_bn_weight, l__mod___features_conv4_1_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_49 = l__mod___features_conv4_1_c3x3_b_bn_running_mean = l__mod___features_conv4_1_c3x3_b_bn_running_var = l__mod___features_conv4_1_c3x3_b_bn_weight = l__mod___features_conv4_1_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_126 = self.L__mod___features_conv4_1_c3x3_b_bn_drop(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_127 = self.L__mod___features_conv4_1_c3x3_b_bn_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_50 = self.L__mod___features_conv4_1_c3x3_b_conv(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_1_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_1_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__42 = l__mod___features_conv4_1_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_1_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_c_bn_running_mean = self.L__mod___features_conv4_1_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_1_c1x1_c_bn_running_var = self.L__mod___features_conv4_1_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_1_c1x1_c_bn_weight = self.L__mod___features_conv4_1_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_1_c1x1_c_bn_bias = self.L__mod___features_conv4_1_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_128 = torch.nn.functional.batch_norm(x_in_50, l__mod___features_conv4_1_c1x1_c_bn_running_mean, l__mod___features_conv4_1_c1x1_c_bn_running_var, l__mod___features_conv4_1_c1x1_c_bn_weight, l__mod___features_conv4_1_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_50 = l__mod___features_conv4_1_c1x1_c_bn_running_mean = l__mod___features_conv4_1_c1x1_c_bn_running_var = l__mod___features_conv4_1_c1x1_c_bn_weight = l__mod___features_conv4_1_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_129 = self.L__mod___features_conv4_1_c1x1_c_bn_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_130 = self.L__mod___features_conv4_1_c1x1_c_bn_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_51 = self.L__mod___features_conv4_1_c1x1_c_conv(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_12 = x_in_51[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_12 = x_in_51[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_13 = x_s1_12 + out1_12;  x_s1_12 = out1_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_13 = torch.cat([x_s2_12, out2_12], dim = 1);  x_s2_12 = out2_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_52 = torch.cat((x_s1_13, x_s2_13), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_2_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_2_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__43 = l__mod___features_conv4_2_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_2_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c1x1_a_bn_running_mean = self.L__mod___features_conv4_2_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c1x1_a_bn_running_var = self.L__mod___features_conv4_2_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_2_c1x1_a_bn_weight = self.L__mod___features_conv4_2_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_2_c1x1_a_bn_bias = self.L__mod___features_conv4_2_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_131 = torch.nn.functional.batch_norm(x_in_52, l__mod___features_conv4_2_c1x1_a_bn_running_mean, l__mod___features_conv4_2_c1x1_a_bn_running_var, l__mod___features_conv4_2_c1x1_a_bn_weight, l__mod___features_conv4_2_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_52 = l__mod___features_conv4_2_c1x1_a_bn_running_mean = l__mod___features_conv4_2_c1x1_a_bn_running_var = l__mod___features_conv4_2_c1x1_a_bn_weight = l__mod___features_conv4_2_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_132 = self.L__mod___features_conv4_2_c1x1_a_bn_drop(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_133 = self.L__mod___features_conv4_2_c1x1_a_bn_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_53 = self.L__mod___features_conv4_2_c1x1_a_conv(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_2_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_2_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__44 = l__mod___features_conv4_2_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_2_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c3x3_b_bn_running_mean = self.L__mod___features_conv4_2_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c3x3_b_bn_running_var = self.L__mod___features_conv4_2_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_2_c3x3_b_bn_weight = self.L__mod___features_conv4_2_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_2_c3x3_b_bn_bias = self.L__mod___features_conv4_2_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_in_53, l__mod___features_conv4_2_c3x3_b_bn_running_mean, l__mod___features_conv4_2_c3x3_b_bn_running_var, l__mod___features_conv4_2_c3x3_b_bn_weight, l__mod___features_conv4_2_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_53 = l__mod___features_conv4_2_c3x3_b_bn_running_mean = l__mod___features_conv4_2_c3x3_b_bn_running_var = l__mod___features_conv4_2_c3x3_b_bn_weight = l__mod___features_conv4_2_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.L__mod___features_conv4_2_c3x3_b_bn_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_136 = self.L__mod___features_conv4_2_c3x3_b_bn_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_54 = self.L__mod___features_conv4_2_c3x3_b_conv(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_2_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_2_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__45 = l__mod___features_conv4_2_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_2_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c1x1_c_bn_running_mean = self.L__mod___features_conv4_2_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_2_c1x1_c_bn_running_var = self.L__mod___features_conv4_2_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_2_c1x1_c_bn_weight = self.L__mod___features_conv4_2_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_2_c1x1_c_bn_bias = self.L__mod___features_conv4_2_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_137 = torch.nn.functional.batch_norm(x_in_54, l__mod___features_conv4_2_c1x1_c_bn_running_mean, l__mod___features_conv4_2_c1x1_c_bn_running_var, l__mod___features_conv4_2_c1x1_c_bn_weight, l__mod___features_conv4_2_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_54 = l__mod___features_conv4_2_c1x1_c_bn_running_mean = l__mod___features_conv4_2_c1x1_c_bn_running_var = l__mod___features_conv4_2_c1x1_c_bn_weight = l__mod___features_conv4_2_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_138 = self.L__mod___features_conv4_2_c1x1_c_bn_drop(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_139 = self.L__mod___features_conv4_2_c1x1_c_bn_act(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_55 = self.L__mod___features_conv4_2_c1x1_c_conv(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_13 = x_in_55[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_13 = x_in_55[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_14 = x_s1_13 + out1_13;  x_s1_13 = out1_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_14 = torch.cat([x_s2_13, out2_13], dim = 1);  x_s2_13 = out2_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_56 = torch.cat((x_s1_14, x_s2_14), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_3_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_3_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__46 = l__mod___features_conv4_3_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_3_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c1x1_a_bn_running_mean = self.L__mod___features_conv4_3_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c1x1_a_bn_running_var = self.L__mod___features_conv4_3_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_3_c1x1_a_bn_weight = self.L__mod___features_conv4_3_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_3_c1x1_a_bn_bias = self.L__mod___features_conv4_3_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_in_56, l__mod___features_conv4_3_c1x1_a_bn_running_mean, l__mod___features_conv4_3_c1x1_a_bn_running_var, l__mod___features_conv4_3_c1x1_a_bn_weight, l__mod___features_conv4_3_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_56 = l__mod___features_conv4_3_c1x1_a_bn_running_mean = l__mod___features_conv4_3_c1x1_a_bn_running_var = l__mod___features_conv4_3_c1x1_a_bn_weight = l__mod___features_conv4_3_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.L__mod___features_conv4_3_c1x1_a_bn_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_142 = self.L__mod___features_conv4_3_c1x1_a_bn_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_57 = self.L__mod___features_conv4_3_c1x1_a_conv(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_3_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_3_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__47 = l__mod___features_conv4_3_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_3_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c3x3_b_bn_running_mean = self.L__mod___features_conv4_3_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c3x3_b_bn_running_var = self.L__mod___features_conv4_3_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_3_c3x3_b_bn_weight = self.L__mod___features_conv4_3_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_3_c3x3_b_bn_bias = self.L__mod___features_conv4_3_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_143 = torch.nn.functional.batch_norm(x_in_57, l__mod___features_conv4_3_c3x3_b_bn_running_mean, l__mod___features_conv4_3_c3x3_b_bn_running_var, l__mod___features_conv4_3_c3x3_b_bn_weight, l__mod___features_conv4_3_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_57 = l__mod___features_conv4_3_c3x3_b_bn_running_mean = l__mod___features_conv4_3_c3x3_b_bn_running_var = l__mod___features_conv4_3_c3x3_b_bn_weight = l__mod___features_conv4_3_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_144 = self.L__mod___features_conv4_3_c3x3_b_bn_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_145 = self.L__mod___features_conv4_3_c3x3_b_bn_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_58 = self.L__mod___features_conv4_3_c3x3_b_conv(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_3_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_3_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__48 = l__mod___features_conv4_3_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_3_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c1x1_c_bn_running_mean = self.L__mod___features_conv4_3_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_3_c1x1_c_bn_running_var = self.L__mod___features_conv4_3_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_3_c1x1_c_bn_weight = self.L__mod___features_conv4_3_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_3_c1x1_c_bn_bias = self.L__mod___features_conv4_3_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_146 = torch.nn.functional.batch_norm(x_in_58, l__mod___features_conv4_3_c1x1_c_bn_running_mean, l__mod___features_conv4_3_c1x1_c_bn_running_var, l__mod___features_conv4_3_c1x1_c_bn_weight, l__mod___features_conv4_3_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_58 = l__mod___features_conv4_3_c1x1_c_bn_running_mean = l__mod___features_conv4_3_c1x1_c_bn_running_var = l__mod___features_conv4_3_c1x1_c_bn_weight = l__mod___features_conv4_3_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_147 = self.L__mod___features_conv4_3_c1x1_c_bn_drop(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_148 = self.L__mod___features_conv4_3_c1x1_c_bn_act(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_59 = self.L__mod___features_conv4_3_c1x1_c_conv(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_14 = x_in_59[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_14 = x_in_59[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_15 = x_s1_14 + out1_14;  x_s1_14 = out1_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_15 = torch.cat([x_s2_14, out2_14], dim = 1);  x_s2_14 = out2_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_60 = torch.cat((x_s1_15, x_s2_15), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_4_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_4_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__49 = l__mod___features_conv4_4_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_4_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c1x1_a_bn_running_mean = self.L__mod___features_conv4_4_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c1x1_a_bn_running_var = self.L__mod___features_conv4_4_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_4_c1x1_a_bn_weight = self.L__mod___features_conv4_4_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_4_c1x1_a_bn_bias = self.L__mod___features_conv4_4_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_149 = torch.nn.functional.batch_norm(x_in_60, l__mod___features_conv4_4_c1x1_a_bn_running_mean, l__mod___features_conv4_4_c1x1_a_bn_running_var, l__mod___features_conv4_4_c1x1_a_bn_weight, l__mod___features_conv4_4_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_60 = l__mod___features_conv4_4_c1x1_a_bn_running_mean = l__mod___features_conv4_4_c1x1_a_bn_running_var = l__mod___features_conv4_4_c1x1_a_bn_weight = l__mod___features_conv4_4_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_150 = self.L__mod___features_conv4_4_c1x1_a_bn_drop(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_151 = self.L__mod___features_conv4_4_c1x1_a_bn_act(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_61 = self.L__mod___features_conv4_4_c1x1_a_conv(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_4_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_4_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__50 = l__mod___features_conv4_4_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_4_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c3x3_b_bn_running_mean = self.L__mod___features_conv4_4_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c3x3_b_bn_running_var = self.L__mod___features_conv4_4_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_4_c3x3_b_bn_weight = self.L__mod___features_conv4_4_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_4_c3x3_b_bn_bias = self.L__mod___features_conv4_4_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_152 = torch.nn.functional.batch_norm(x_in_61, l__mod___features_conv4_4_c3x3_b_bn_running_mean, l__mod___features_conv4_4_c3x3_b_bn_running_var, l__mod___features_conv4_4_c3x3_b_bn_weight, l__mod___features_conv4_4_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_61 = l__mod___features_conv4_4_c3x3_b_bn_running_mean = l__mod___features_conv4_4_c3x3_b_bn_running_var = l__mod___features_conv4_4_c3x3_b_bn_weight = l__mod___features_conv4_4_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_153 = self.L__mod___features_conv4_4_c3x3_b_bn_drop(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.L__mod___features_conv4_4_c3x3_b_bn_act(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_62 = self.L__mod___features_conv4_4_c3x3_b_conv(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_4_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_4_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__51 = l__mod___features_conv4_4_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_4_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c1x1_c_bn_running_mean = self.L__mod___features_conv4_4_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_4_c1x1_c_bn_running_var = self.L__mod___features_conv4_4_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_4_c1x1_c_bn_weight = self.L__mod___features_conv4_4_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_4_c1x1_c_bn_bias = self.L__mod___features_conv4_4_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_155 = torch.nn.functional.batch_norm(x_in_62, l__mod___features_conv4_4_c1x1_c_bn_running_mean, l__mod___features_conv4_4_c1x1_c_bn_running_var, l__mod___features_conv4_4_c1x1_c_bn_weight, l__mod___features_conv4_4_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_62 = l__mod___features_conv4_4_c1x1_c_bn_running_mean = l__mod___features_conv4_4_c1x1_c_bn_running_var = l__mod___features_conv4_4_c1x1_c_bn_weight = l__mod___features_conv4_4_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_156 = self.L__mod___features_conv4_4_c1x1_c_bn_drop(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_157 = self.L__mod___features_conv4_4_c1x1_c_bn_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_63 = self.L__mod___features_conv4_4_c1x1_c_conv(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_15 = x_in_63[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_15 = x_in_63[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_16 = x_s1_15 + out1_15;  x_s1_15 = out1_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_16 = torch.cat([x_s2_15, out2_15], dim = 1);  x_s2_15 = out2_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_64 = torch.cat((x_s1_16, x_s2_16), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_5_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_5_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__52 = l__mod___features_conv4_5_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_5_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c1x1_a_bn_running_mean = self.L__mod___features_conv4_5_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c1x1_a_bn_running_var = self.L__mod___features_conv4_5_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_5_c1x1_a_bn_weight = self.L__mod___features_conv4_5_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_5_c1x1_a_bn_bias = self.L__mod___features_conv4_5_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_158 = torch.nn.functional.batch_norm(x_in_64, l__mod___features_conv4_5_c1x1_a_bn_running_mean, l__mod___features_conv4_5_c1x1_a_bn_running_var, l__mod___features_conv4_5_c1x1_a_bn_weight, l__mod___features_conv4_5_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_64 = l__mod___features_conv4_5_c1x1_a_bn_running_mean = l__mod___features_conv4_5_c1x1_a_bn_running_var = l__mod___features_conv4_5_c1x1_a_bn_weight = l__mod___features_conv4_5_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_159 = self.L__mod___features_conv4_5_c1x1_a_bn_drop(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_160 = self.L__mod___features_conv4_5_c1x1_a_bn_act(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_65 = self.L__mod___features_conv4_5_c1x1_a_conv(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_5_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_5_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__53 = l__mod___features_conv4_5_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_5_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c3x3_b_bn_running_mean = self.L__mod___features_conv4_5_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c3x3_b_bn_running_var = self.L__mod___features_conv4_5_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_5_c3x3_b_bn_weight = self.L__mod___features_conv4_5_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_5_c3x3_b_bn_bias = self.L__mod___features_conv4_5_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_161 = torch.nn.functional.batch_norm(x_in_65, l__mod___features_conv4_5_c3x3_b_bn_running_mean, l__mod___features_conv4_5_c3x3_b_bn_running_var, l__mod___features_conv4_5_c3x3_b_bn_weight, l__mod___features_conv4_5_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_65 = l__mod___features_conv4_5_c3x3_b_bn_running_mean = l__mod___features_conv4_5_c3x3_b_bn_running_var = l__mod___features_conv4_5_c3x3_b_bn_weight = l__mod___features_conv4_5_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_162 = self.L__mod___features_conv4_5_c3x3_b_bn_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_163 = self.L__mod___features_conv4_5_c3x3_b_bn_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_66 = self.L__mod___features_conv4_5_c3x3_b_conv(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_5_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_5_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__54 = l__mod___features_conv4_5_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_5_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c1x1_c_bn_running_mean = self.L__mod___features_conv4_5_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_5_c1x1_c_bn_running_var = self.L__mod___features_conv4_5_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_5_c1x1_c_bn_weight = self.L__mod___features_conv4_5_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_5_c1x1_c_bn_bias = self.L__mod___features_conv4_5_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_164 = torch.nn.functional.batch_norm(x_in_66, l__mod___features_conv4_5_c1x1_c_bn_running_mean, l__mod___features_conv4_5_c1x1_c_bn_running_var, l__mod___features_conv4_5_c1x1_c_bn_weight, l__mod___features_conv4_5_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_66 = l__mod___features_conv4_5_c1x1_c_bn_running_mean = l__mod___features_conv4_5_c1x1_c_bn_running_var = l__mod___features_conv4_5_c1x1_c_bn_weight = l__mod___features_conv4_5_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_165 = self.L__mod___features_conv4_5_c1x1_c_bn_drop(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_166 = self.L__mod___features_conv4_5_c1x1_c_bn_act(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_67 = self.L__mod___features_conv4_5_c1x1_c_conv(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_16 = x_in_67[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_16 = x_in_67[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_17 = x_s1_16 + out1_16;  x_s1_16 = out1_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_17 = torch.cat([x_s2_16, out2_16], dim = 1);  x_s2_16 = out2_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_68 = torch.cat((x_s1_17, x_s2_17), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_6_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_6_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__55 = l__mod___features_conv4_6_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_6_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c1x1_a_bn_running_mean = self.L__mod___features_conv4_6_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c1x1_a_bn_running_var = self.L__mod___features_conv4_6_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_6_c1x1_a_bn_weight = self.L__mod___features_conv4_6_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_6_c1x1_a_bn_bias = self.L__mod___features_conv4_6_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_167 = torch.nn.functional.batch_norm(x_in_68, l__mod___features_conv4_6_c1x1_a_bn_running_mean, l__mod___features_conv4_6_c1x1_a_bn_running_var, l__mod___features_conv4_6_c1x1_a_bn_weight, l__mod___features_conv4_6_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_68 = l__mod___features_conv4_6_c1x1_a_bn_running_mean = l__mod___features_conv4_6_c1x1_a_bn_running_var = l__mod___features_conv4_6_c1x1_a_bn_weight = l__mod___features_conv4_6_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_168 = self.L__mod___features_conv4_6_c1x1_a_bn_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_169 = self.L__mod___features_conv4_6_c1x1_a_bn_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_69 = self.L__mod___features_conv4_6_c1x1_a_conv(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_6_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_6_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__56 = l__mod___features_conv4_6_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_6_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c3x3_b_bn_running_mean = self.L__mod___features_conv4_6_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c3x3_b_bn_running_var = self.L__mod___features_conv4_6_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_6_c3x3_b_bn_weight = self.L__mod___features_conv4_6_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_6_c3x3_b_bn_bias = self.L__mod___features_conv4_6_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_170 = torch.nn.functional.batch_norm(x_in_69, l__mod___features_conv4_6_c3x3_b_bn_running_mean, l__mod___features_conv4_6_c3x3_b_bn_running_var, l__mod___features_conv4_6_c3x3_b_bn_weight, l__mod___features_conv4_6_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_69 = l__mod___features_conv4_6_c3x3_b_bn_running_mean = l__mod___features_conv4_6_c3x3_b_bn_running_var = l__mod___features_conv4_6_c3x3_b_bn_weight = l__mod___features_conv4_6_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_171 = self.L__mod___features_conv4_6_c3x3_b_bn_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_172 = self.L__mod___features_conv4_6_c3x3_b_bn_act(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_70 = self.L__mod___features_conv4_6_c3x3_b_conv(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_6_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_6_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__57 = l__mod___features_conv4_6_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_6_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c1x1_c_bn_running_mean = self.L__mod___features_conv4_6_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_6_c1x1_c_bn_running_var = self.L__mod___features_conv4_6_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_6_c1x1_c_bn_weight = self.L__mod___features_conv4_6_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_6_c1x1_c_bn_bias = self.L__mod___features_conv4_6_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_in_70, l__mod___features_conv4_6_c1x1_c_bn_running_mean, l__mod___features_conv4_6_c1x1_c_bn_running_var, l__mod___features_conv4_6_c1x1_c_bn_weight, l__mod___features_conv4_6_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_70 = l__mod___features_conv4_6_c1x1_c_bn_running_mean = l__mod___features_conv4_6_c1x1_c_bn_running_var = l__mod___features_conv4_6_c1x1_c_bn_weight = l__mod___features_conv4_6_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.L__mod___features_conv4_6_c1x1_c_bn_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_175 = self.L__mod___features_conv4_6_c1x1_c_bn_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_71 = self.L__mod___features_conv4_6_c1x1_c_conv(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_17 = x_in_71[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_17 = x_in_71[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_18 = x_s1_17 + out1_17;  x_s1_17 = out1_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_18 = torch.cat([x_s2_17, out2_17], dim = 1);  x_s2_17 = out2_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_72 = torch.cat((x_s1_18, x_s2_18), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_7_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_7_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__58 = l__mod___features_conv4_7_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_7_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c1x1_a_bn_running_mean = self.L__mod___features_conv4_7_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c1x1_a_bn_running_var = self.L__mod___features_conv4_7_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_7_c1x1_a_bn_weight = self.L__mod___features_conv4_7_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_7_c1x1_a_bn_bias = self.L__mod___features_conv4_7_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_176 = torch.nn.functional.batch_norm(x_in_72, l__mod___features_conv4_7_c1x1_a_bn_running_mean, l__mod___features_conv4_7_c1x1_a_bn_running_var, l__mod___features_conv4_7_c1x1_a_bn_weight, l__mod___features_conv4_7_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_72 = l__mod___features_conv4_7_c1x1_a_bn_running_mean = l__mod___features_conv4_7_c1x1_a_bn_running_var = l__mod___features_conv4_7_c1x1_a_bn_weight = l__mod___features_conv4_7_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_177 = self.L__mod___features_conv4_7_c1x1_a_bn_drop(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_178 = self.L__mod___features_conv4_7_c1x1_a_bn_act(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_73 = self.L__mod___features_conv4_7_c1x1_a_conv(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_7_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_7_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__59 = l__mod___features_conv4_7_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_7_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c3x3_b_bn_running_mean = self.L__mod___features_conv4_7_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c3x3_b_bn_running_var = self.L__mod___features_conv4_7_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_7_c3x3_b_bn_weight = self.L__mod___features_conv4_7_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_7_c3x3_b_bn_bias = self.L__mod___features_conv4_7_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_179 = torch.nn.functional.batch_norm(x_in_73, l__mod___features_conv4_7_c3x3_b_bn_running_mean, l__mod___features_conv4_7_c3x3_b_bn_running_var, l__mod___features_conv4_7_c3x3_b_bn_weight, l__mod___features_conv4_7_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_73 = l__mod___features_conv4_7_c3x3_b_bn_running_mean = l__mod___features_conv4_7_c3x3_b_bn_running_var = l__mod___features_conv4_7_c3x3_b_bn_weight = l__mod___features_conv4_7_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_180 = self.L__mod___features_conv4_7_c3x3_b_bn_drop(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_181 = self.L__mod___features_conv4_7_c3x3_b_bn_act(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_74 = self.L__mod___features_conv4_7_c3x3_b_conv(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_7_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_7_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__60 = l__mod___features_conv4_7_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_7_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c1x1_c_bn_running_mean = self.L__mod___features_conv4_7_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_7_c1x1_c_bn_running_var = self.L__mod___features_conv4_7_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_7_c1x1_c_bn_weight = self.L__mod___features_conv4_7_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_7_c1x1_c_bn_bias = self.L__mod___features_conv4_7_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_182 = torch.nn.functional.batch_norm(x_in_74, l__mod___features_conv4_7_c1x1_c_bn_running_mean, l__mod___features_conv4_7_c1x1_c_bn_running_var, l__mod___features_conv4_7_c1x1_c_bn_weight, l__mod___features_conv4_7_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_74 = l__mod___features_conv4_7_c1x1_c_bn_running_mean = l__mod___features_conv4_7_c1x1_c_bn_running_var = l__mod___features_conv4_7_c1x1_c_bn_weight = l__mod___features_conv4_7_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_183 = self.L__mod___features_conv4_7_c1x1_c_bn_drop(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_184 = self.L__mod___features_conv4_7_c1x1_c_bn_act(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_75 = self.L__mod___features_conv4_7_c1x1_c_conv(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_18 = x_in_75[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_18 = x_in_75[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_19 = x_s1_18 + out1_18;  x_s1_18 = out1_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_19 = torch.cat([x_s2_18, out2_18], dim = 1);  x_s2_18 = out2_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_76 = torch.cat((x_s1_19, x_s2_19), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_8_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_8_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__61 = l__mod___features_conv4_8_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_8_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c1x1_a_bn_running_mean = self.L__mod___features_conv4_8_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c1x1_a_bn_running_var = self.L__mod___features_conv4_8_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_8_c1x1_a_bn_weight = self.L__mod___features_conv4_8_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_8_c1x1_a_bn_bias = self.L__mod___features_conv4_8_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_185 = torch.nn.functional.batch_norm(x_in_76, l__mod___features_conv4_8_c1x1_a_bn_running_mean, l__mod___features_conv4_8_c1x1_a_bn_running_var, l__mod___features_conv4_8_c1x1_a_bn_weight, l__mod___features_conv4_8_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_76 = l__mod___features_conv4_8_c1x1_a_bn_running_mean = l__mod___features_conv4_8_c1x1_a_bn_running_var = l__mod___features_conv4_8_c1x1_a_bn_weight = l__mod___features_conv4_8_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_186 = self.L__mod___features_conv4_8_c1x1_a_bn_drop(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.L__mod___features_conv4_8_c1x1_a_bn_act(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_77 = self.L__mod___features_conv4_8_c1x1_a_conv(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_8_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_8_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__62 = l__mod___features_conv4_8_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_8_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c3x3_b_bn_running_mean = self.L__mod___features_conv4_8_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c3x3_b_bn_running_var = self.L__mod___features_conv4_8_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_8_c3x3_b_bn_weight = self.L__mod___features_conv4_8_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_8_c3x3_b_bn_bias = self.L__mod___features_conv4_8_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_188 = torch.nn.functional.batch_norm(x_in_77, l__mod___features_conv4_8_c3x3_b_bn_running_mean, l__mod___features_conv4_8_c3x3_b_bn_running_var, l__mod___features_conv4_8_c3x3_b_bn_weight, l__mod___features_conv4_8_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_77 = l__mod___features_conv4_8_c3x3_b_bn_running_mean = l__mod___features_conv4_8_c3x3_b_bn_running_var = l__mod___features_conv4_8_c3x3_b_bn_weight = l__mod___features_conv4_8_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_189 = self.L__mod___features_conv4_8_c3x3_b_bn_drop(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_190 = self.L__mod___features_conv4_8_c3x3_b_bn_act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_78 = self.L__mod___features_conv4_8_c3x3_b_conv(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_8_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_8_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__63 = l__mod___features_conv4_8_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_8_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c1x1_c_bn_running_mean = self.L__mod___features_conv4_8_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_8_c1x1_c_bn_running_var = self.L__mod___features_conv4_8_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_8_c1x1_c_bn_weight = self.L__mod___features_conv4_8_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_8_c1x1_c_bn_bias = self.L__mod___features_conv4_8_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_191 = torch.nn.functional.batch_norm(x_in_78, l__mod___features_conv4_8_c1x1_c_bn_running_mean, l__mod___features_conv4_8_c1x1_c_bn_running_var, l__mod___features_conv4_8_c1x1_c_bn_weight, l__mod___features_conv4_8_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_78 = l__mod___features_conv4_8_c1x1_c_bn_running_mean = l__mod___features_conv4_8_c1x1_c_bn_running_var = l__mod___features_conv4_8_c1x1_c_bn_weight = l__mod___features_conv4_8_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_192 = self.L__mod___features_conv4_8_c1x1_c_bn_drop(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_193 = self.L__mod___features_conv4_8_c1x1_c_bn_act(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_79 = self.L__mod___features_conv4_8_c1x1_c_conv(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_19 = x_in_79[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_19 = x_in_79[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_20 = x_s1_19 + out1_19;  x_s1_19 = out1_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_20 = torch.cat([x_s2_19, out2_19], dim = 1);  x_s2_19 = out2_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_80 = torch.cat((x_s1_20, x_s2_20), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_9_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_9_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__64 = l__mod___features_conv4_9_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_9_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c1x1_a_bn_running_mean = self.L__mod___features_conv4_9_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c1x1_a_bn_running_var = self.L__mod___features_conv4_9_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_9_c1x1_a_bn_weight = self.L__mod___features_conv4_9_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_9_c1x1_a_bn_bias = self.L__mod___features_conv4_9_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_194 = torch.nn.functional.batch_norm(x_in_80, l__mod___features_conv4_9_c1x1_a_bn_running_mean, l__mod___features_conv4_9_c1x1_a_bn_running_var, l__mod___features_conv4_9_c1x1_a_bn_weight, l__mod___features_conv4_9_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_80 = l__mod___features_conv4_9_c1x1_a_bn_running_mean = l__mod___features_conv4_9_c1x1_a_bn_running_var = l__mod___features_conv4_9_c1x1_a_bn_weight = l__mod___features_conv4_9_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_195 = self.L__mod___features_conv4_9_c1x1_a_bn_drop(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_196 = self.L__mod___features_conv4_9_c1x1_a_bn_act(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_81 = self.L__mod___features_conv4_9_c1x1_a_conv(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_9_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_9_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__65 = l__mod___features_conv4_9_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_9_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c3x3_b_bn_running_mean = self.L__mod___features_conv4_9_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c3x3_b_bn_running_var = self.L__mod___features_conv4_9_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_9_c3x3_b_bn_weight = self.L__mod___features_conv4_9_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_9_c3x3_b_bn_bias = self.L__mod___features_conv4_9_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_197 = torch.nn.functional.batch_norm(x_in_81, l__mod___features_conv4_9_c3x3_b_bn_running_mean, l__mod___features_conv4_9_c3x3_b_bn_running_var, l__mod___features_conv4_9_c3x3_b_bn_weight, l__mod___features_conv4_9_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_81 = l__mod___features_conv4_9_c3x3_b_bn_running_mean = l__mod___features_conv4_9_c3x3_b_bn_running_var = l__mod___features_conv4_9_c3x3_b_bn_weight = l__mod___features_conv4_9_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.L__mod___features_conv4_9_c3x3_b_bn_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_199 = self.L__mod___features_conv4_9_c3x3_b_bn_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_82 = self.L__mod___features_conv4_9_c3x3_b_conv(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_9_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_9_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__66 = l__mod___features_conv4_9_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_9_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c1x1_c_bn_running_mean = self.L__mod___features_conv4_9_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_9_c1x1_c_bn_running_var = self.L__mod___features_conv4_9_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_9_c1x1_c_bn_weight = self.L__mod___features_conv4_9_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_9_c1x1_c_bn_bias = self.L__mod___features_conv4_9_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_200 = torch.nn.functional.batch_norm(x_in_82, l__mod___features_conv4_9_c1x1_c_bn_running_mean, l__mod___features_conv4_9_c1x1_c_bn_running_var, l__mod___features_conv4_9_c1x1_c_bn_weight, l__mod___features_conv4_9_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_82 = l__mod___features_conv4_9_c1x1_c_bn_running_mean = l__mod___features_conv4_9_c1x1_c_bn_running_var = l__mod___features_conv4_9_c1x1_c_bn_weight = l__mod___features_conv4_9_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_201 = self.L__mod___features_conv4_9_c1x1_c_bn_drop(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_202 = self.L__mod___features_conv4_9_c1x1_c_bn_act(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_83 = self.L__mod___features_conv4_9_c1x1_c_conv(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_20 = x_in_83[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_20 = x_in_83[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_21 = x_s1_20 + out1_20;  x_s1_20 = out1_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_21 = torch.cat([x_s2_20, out2_20], dim = 1);  x_s2_20 = out2_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_84 = torch.cat((x_s1_21, x_s2_21), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_10_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_10_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__67 = l__mod___features_conv4_10_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_10_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c1x1_a_bn_running_mean = self.L__mod___features_conv4_10_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c1x1_a_bn_running_var = self.L__mod___features_conv4_10_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_10_c1x1_a_bn_weight = self.L__mod___features_conv4_10_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_10_c1x1_a_bn_bias = self.L__mod___features_conv4_10_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_203 = torch.nn.functional.batch_norm(x_in_84, l__mod___features_conv4_10_c1x1_a_bn_running_mean, l__mod___features_conv4_10_c1x1_a_bn_running_var, l__mod___features_conv4_10_c1x1_a_bn_weight, l__mod___features_conv4_10_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_84 = l__mod___features_conv4_10_c1x1_a_bn_running_mean = l__mod___features_conv4_10_c1x1_a_bn_running_var = l__mod___features_conv4_10_c1x1_a_bn_weight = l__mod___features_conv4_10_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_204 = self.L__mod___features_conv4_10_c1x1_a_bn_drop(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_205 = self.L__mod___features_conv4_10_c1x1_a_bn_act(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_85 = self.L__mod___features_conv4_10_c1x1_a_conv(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_10_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_10_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__68 = l__mod___features_conv4_10_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_10_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c3x3_b_bn_running_mean = self.L__mod___features_conv4_10_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c3x3_b_bn_running_var = self.L__mod___features_conv4_10_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_10_c3x3_b_bn_weight = self.L__mod___features_conv4_10_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_10_c3x3_b_bn_bias = self.L__mod___features_conv4_10_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_in_85, l__mod___features_conv4_10_c3x3_b_bn_running_mean, l__mod___features_conv4_10_c3x3_b_bn_running_var, l__mod___features_conv4_10_c3x3_b_bn_weight, l__mod___features_conv4_10_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_85 = l__mod___features_conv4_10_c3x3_b_bn_running_mean = l__mod___features_conv4_10_c3x3_b_bn_running_var = l__mod___features_conv4_10_c3x3_b_bn_weight = l__mod___features_conv4_10_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.L__mod___features_conv4_10_c3x3_b_bn_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_208 = self.L__mod___features_conv4_10_c3x3_b_bn_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_86 = self.L__mod___features_conv4_10_c3x3_b_conv(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_10_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_10_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__69 = l__mod___features_conv4_10_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_10_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c1x1_c_bn_running_mean = self.L__mod___features_conv4_10_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_10_c1x1_c_bn_running_var = self.L__mod___features_conv4_10_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_10_c1x1_c_bn_weight = self.L__mod___features_conv4_10_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_10_c1x1_c_bn_bias = self.L__mod___features_conv4_10_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_209 = torch.nn.functional.batch_norm(x_in_86, l__mod___features_conv4_10_c1x1_c_bn_running_mean, l__mod___features_conv4_10_c1x1_c_bn_running_var, l__mod___features_conv4_10_c1x1_c_bn_weight, l__mod___features_conv4_10_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_86 = l__mod___features_conv4_10_c1x1_c_bn_running_mean = l__mod___features_conv4_10_c1x1_c_bn_running_var = l__mod___features_conv4_10_c1x1_c_bn_weight = l__mod___features_conv4_10_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_210 = self.L__mod___features_conv4_10_c1x1_c_bn_drop(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_211 = self.L__mod___features_conv4_10_c1x1_c_bn_act(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_87 = self.L__mod___features_conv4_10_c1x1_c_conv(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_21 = x_in_87[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_21 = x_in_87[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_22 = x_s1_21 + out1_21;  x_s1_21 = out1_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_22 = torch.cat([x_s2_21, out2_21], dim = 1);  x_s2_21 = out2_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_88 = torch.cat((x_s1_22, x_s2_22), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_11_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_11_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__70 = l__mod___features_conv4_11_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_11_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c1x1_a_bn_running_mean = self.L__mod___features_conv4_11_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c1x1_a_bn_running_var = self.L__mod___features_conv4_11_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_11_c1x1_a_bn_weight = self.L__mod___features_conv4_11_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_11_c1x1_a_bn_bias = self.L__mod___features_conv4_11_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_212 = torch.nn.functional.batch_norm(x_in_88, l__mod___features_conv4_11_c1x1_a_bn_running_mean, l__mod___features_conv4_11_c1x1_a_bn_running_var, l__mod___features_conv4_11_c1x1_a_bn_weight, l__mod___features_conv4_11_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_88 = l__mod___features_conv4_11_c1x1_a_bn_running_mean = l__mod___features_conv4_11_c1x1_a_bn_running_var = l__mod___features_conv4_11_c1x1_a_bn_weight = l__mod___features_conv4_11_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_213 = self.L__mod___features_conv4_11_c1x1_a_bn_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_214 = self.L__mod___features_conv4_11_c1x1_a_bn_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_89 = self.L__mod___features_conv4_11_c1x1_a_conv(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_11_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_11_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__71 = l__mod___features_conv4_11_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_11_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c3x3_b_bn_running_mean = self.L__mod___features_conv4_11_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c3x3_b_bn_running_var = self.L__mod___features_conv4_11_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_11_c3x3_b_bn_weight = self.L__mod___features_conv4_11_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_11_c3x3_b_bn_bias = self.L__mod___features_conv4_11_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_215 = torch.nn.functional.batch_norm(x_in_89, l__mod___features_conv4_11_c3x3_b_bn_running_mean, l__mod___features_conv4_11_c3x3_b_bn_running_var, l__mod___features_conv4_11_c3x3_b_bn_weight, l__mod___features_conv4_11_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_89 = l__mod___features_conv4_11_c3x3_b_bn_running_mean = l__mod___features_conv4_11_c3x3_b_bn_running_var = l__mod___features_conv4_11_c3x3_b_bn_weight = l__mod___features_conv4_11_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_216 = self.L__mod___features_conv4_11_c3x3_b_bn_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_217 = self.L__mod___features_conv4_11_c3x3_b_bn_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_90 = self.L__mod___features_conv4_11_c3x3_b_conv(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_11_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_11_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__72 = l__mod___features_conv4_11_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_11_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c1x1_c_bn_running_mean = self.L__mod___features_conv4_11_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_11_c1x1_c_bn_running_var = self.L__mod___features_conv4_11_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_11_c1x1_c_bn_weight = self.L__mod___features_conv4_11_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_11_c1x1_c_bn_bias = self.L__mod___features_conv4_11_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_218 = torch.nn.functional.batch_norm(x_in_90, l__mod___features_conv4_11_c1x1_c_bn_running_mean, l__mod___features_conv4_11_c1x1_c_bn_running_var, l__mod___features_conv4_11_c1x1_c_bn_weight, l__mod___features_conv4_11_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_90 = l__mod___features_conv4_11_c1x1_c_bn_running_mean = l__mod___features_conv4_11_c1x1_c_bn_running_var = l__mod___features_conv4_11_c1x1_c_bn_weight = l__mod___features_conv4_11_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_219 = self.L__mod___features_conv4_11_c1x1_c_bn_drop(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_220 = self.L__mod___features_conv4_11_c1x1_c_bn_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_91 = self.L__mod___features_conv4_11_c1x1_c_conv(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_22 = x_in_91[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_22 = x_in_91[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_23 = x_s1_22 + out1_22;  x_s1_22 = out1_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_23 = torch.cat([x_s2_22, out2_22], dim = 1);  x_s2_22 = out2_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_92 = torch.cat((x_s1_23, x_s2_23), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_12_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_12_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__73 = l__mod___features_conv4_12_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_12_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c1x1_a_bn_running_mean = self.L__mod___features_conv4_12_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c1x1_a_bn_running_var = self.L__mod___features_conv4_12_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_12_c1x1_a_bn_weight = self.L__mod___features_conv4_12_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_12_c1x1_a_bn_bias = self.L__mod___features_conv4_12_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_221 = torch.nn.functional.batch_norm(x_in_92, l__mod___features_conv4_12_c1x1_a_bn_running_mean, l__mod___features_conv4_12_c1x1_a_bn_running_var, l__mod___features_conv4_12_c1x1_a_bn_weight, l__mod___features_conv4_12_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_92 = l__mod___features_conv4_12_c1x1_a_bn_running_mean = l__mod___features_conv4_12_c1x1_a_bn_running_var = l__mod___features_conv4_12_c1x1_a_bn_weight = l__mod___features_conv4_12_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_222 = self.L__mod___features_conv4_12_c1x1_a_bn_drop(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_223 = self.L__mod___features_conv4_12_c1x1_a_bn_act(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_93 = self.L__mod___features_conv4_12_c1x1_a_conv(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_12_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_12_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__74 = l__mod___features_conv4_12_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_12_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c3x3_b_bn_running_mean = self.L__mod___features_conv4_12_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c3x3_b_bn_running_var = self.L__mod___features_conv4_12_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_12_c3x3_b_bn_weight = self.L__mod___features_conv4_12_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_12_c3x3_b_bn_bias = self.L__mod___features_conv4_12_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_224 = torch.nn.functional.batch_norm(x_in_93, l__mod___features_conv4_12_c3x3_b_bn_running_mean, l__mod___features_conv4_12_c3x3_b_bn_running_var, l__mod___features_conv4_12_c3x3_b_bn_weight, l__mod___features_conv4_12_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_93 = l__mod___features_conv4_12_c3x3_b_bn_running_mean = l__mod___features_conv4_12_c3x3_b_bn_running_var = l__mod___features_conv4_12_c3x3_b_bn_weight = l__mod___features_conv4_12_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_225 = self.L__mod___features_conv4_12_c3x3_b_bn_drop(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_226 = self.L__mod___features_conv4_12_c3x3_b_bn_act(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_94 = self.L__mod___features_conv4_12_c3x3_b_conv(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_12_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_12_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__75 = l__mod___features_conv4_12_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_12_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c1x1_c_bn_running_mean = self.L__mod___features_conv4_12_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_12_c1x1_c_bn_running_var = self.L__mod___features_conv4_12_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_12_c1x1_c_bn_weight = self.L__mod___features_conv4_12_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_12_c1x1_c_bn_bias = self.L__mod___features_conv4_12_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_227 = torch.nn.functional.batch_norm(x_in_94, l__mod___features_conv4_12_c1x1_c_bn_running_mean, l__mod___features_conv4_12_c1x1_c_bn_running_var, l__mod___features_conv4_12_c1x1_c_bn_weight, l__mod___features_conv4_12_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_94 = l__mod___features_conv4_12_c1x1_c_bn_running_mean = l__mod___features_conv4_12_c1x1_c_bn_running_var = l__mod___features_conv4_12_c1x1_c_bn_weight = l__mod___features_conv4_12_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_228 = self.L__mod___features_conv4_12_c1x1_c_bn_drop(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_229 = self.L__mod___features_conv4_12_c1x1_c_bn_act(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_95 = self.L__mod___features_conv4_12_c1x1_c_conv(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_23 = x_in_95[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_23 = x_in_95[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_24 = x_s1_23 + out1_23;  x_s1_23 = out1_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_24 = torch.cat([x_s2_23, out2_23], dim = 1);  x_s2_23 = out2_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_96 = torch.cat((x_s1_24, x_s2_24), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_13_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_13_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__76 = l__mod___features_conv4_13_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_13_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c1x1_a_bn_running_mean = self.L__mod___features_conv4_13_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c1x1_a_bn_running_var = self.L__mod___features_conv4_13_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_13_c1x1_a_bn_weight = self.L__mod___features_conv4_13_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_13_c1x1_a_bn_bias = self.L__mod___features_conv4_13_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_230 = torch.nn.functional.batch_norm(x_in_96, l__mod___features_conv4_13_c1x1_a_bn_running_mean, l__mod___features_conv4_13_c1x1_a_bn_running_var, l__mod___features_conv4_13_c1x1_a_bn_weight, l__mod___features_conv4_13_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_96 = l__mod___features_conv4_13_c1x1_a_bn_running_mean = l__mod___features_conv4_13_c1x1_a_bn_running_var = l__mod___features_conv4_13_c1x1_a_bn_weight = l__mod___features_conv4_13_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_231 = self.L__mod___features_conv4_13_c1x1_a_bn_drop(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_232 = self.L__mod___features_conv4_13_c1x1_a_bn_act(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_97 = self.L__mod___features_conv4_13_c1x1_a_conv(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_13_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_13_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__77 = l__mod___features_conv4_13_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_13_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c3x3_b_bn_running_mean = self.L__mod___features_conv4_13_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c3x3_b_bn_running_var = self.L__mod___features_conv4_13_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_13_c3x3_b_bn_weight = self.L__mod___features_conv4_13_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_13_c3x3_b_bn_bias = self.L__mod___features_conv4_13_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_233 = torch.nn.functional.batch_norm(x_in_97, l__mod___features_conv4_13_c3x3_b_bn_running_mean, l__mod___features_conv4_13_c3x3_b_bn_running_var, l__mod___features_conv4_13_c3x3_b_bn_weight, l__mod___features_conv4_13_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_97 = l__mod___features_conv4_13_c3x3_b_bn_running_mean = l__mod___features_conv4_13_c3x3_b_bn_running_var = l__mod___features_conv4_13_c3x3_b_bn_weight = l__mod___features_conv4_13_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_234 = self.L__mod___features_conv4_13_c3x3_b_bn_drop(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_235 = self.L__mod___features_conv4_13_c3x3_b_bn_act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_98 = self.L__mod___features_conv4_13_c3x3_b_conv(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_13_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_13_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__78 = l__mod___features_conv4_13_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_13_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c1x1_c_bn_running_mean = self.L__mod___features_conv4_13_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_13_c1x1_c_bn_running_var = self.L__mod___features_conv4_13_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_13_c1x1_c_bn_weight = self.L__mod___features_conv4_13_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_13_c1x1_c_bn_bias = self.L__mod___features_conv4_13_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_236 = torch.nn.functional.batch_norm(x_in_98, l__mod___features_conv4_13_c1x1_c_bn_running_mean, l__mod___features_conv4_13_c1x1_c_bn_running_var, l__mod___features_conv4_13_c1x1_c_bn_weight, l__mod___features_conv4_13_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_98 = l__mod___features_conv4_13_c1x1_c_bn_running_mean = l__mod___features_conv4_13_c1x1_c_bn_running_var = l__mod___features_conv4_13_c1x1_c_bn_weight = l__mod___features_conv4_13_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_237 = self.L__mod___features_conv4_13_c1x1_c_bn_drop(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_238 = self.L__mod___features_conv4_13_c1x1_c_bn_act(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_99 = self.L__mod___features_conv4_13_c1x1_c_conv(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_24 = x_in_99[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_24 = x_in_99[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_25 = x_s1_24 + out1_24;  x_s1_24 = out1_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_25 = torch.cat([x_s2_24, out2_24], dim = 1);  x_s2_24 = out2_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_100 = torch.cat((x_s1_25, x_s2_25), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_14_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_14_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__79 = l__mod___features_conv4_14_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_14_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c1x1_a_bn_running_mean = self.L__mod___features_conv4_14_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c1x1_a_bn_running_var = self.L__mod___features_conv4_14_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_14_c1x1_a_bn_weight = self.L__mod___features_conv4_14_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_14_c1x1_a_bn_bias = self.L__mod___features_conv4_14_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_239 = torch.nn.functional.batch_norm(x_in_100, l__mod___features_conv4_14_c1x1_a_bn_running_mean, l__mod___features_conv4_14_c1x1_a_bn_running_var, l__mod___features_conv4_14_c1x1_a_bn_weight, l__mod___features_conv4_14_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_100 = l__mod___features_conv4_14_c1x1_a_bn_running_mean = l__mod___features_conv4_14_c1x1_a_bn_running_var = l__mod___features_conv4_14_c1x1_a_bn_weight = l__mod___features_conv4_14_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_240 = self.L__mod___features_conv4_14_c1x1_a_bn_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_241 = self.L__mod___features_conv4_14_c1x1_a_bn_act(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_101 = self.L__mod___features_conv4_14_c1x1_a_conv(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_14_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_14_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__80 = l__mod___features_conv4_14_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_14_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c3x3_b_bn_running_mean = self.L__mod___features_conv4_14_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c3x3_b_bn_running_var = self.L__mod___features_conv4_14_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_14_c3x3_b_bn_weight = self.L__mod___features_conv4_14_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_14_c3x3_b_bn_bias = self.L__mod___features_conv4_14_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_242 = torch.nn.functional.batch_norm(x_in_101, l__mod___features_conv4_14_c3x3_b_bn_running_mean, l__mod___features_conv4_14_c3x3_b_bn_running_var, l__mod___features_conv4_14_c3x3_b_bn_weight, l__mod___features_conv4_14_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_101 = l__mod___features_conv4_14_c3x3_b_bn_running_mean = l__mod___features_conv4_14_c3x3_b_bn_running_var = l__mod___features_conv4_14_c3x3_b_bn_weight = l__mod___features_conv4_14_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_243 = self.L__mod___features_conv4_14_c3x3_b_bn_drop(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_244 = self.L__mod___features_conv4_14_c3x3_b_bn_act(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_102 = self.L__mod___features_conv4_14_c3x3_b_conv(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_14_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_14_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__81 = l__mod___features_conv4_14_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_14_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c1x1_c_bn_running_mean = self.L__mod___features_conv4_14_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_14_c1x1_c_bn_running_var = self.L__mod___features_conv4_14_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_14_c1x1_c_bn_weight = self.L__mod___features_conv4_14_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_14_c1x1_c_bn_bias = self.L__mod___features_conv4_14_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_245 = torch.nn.functional.batch_norm(x_in_102, l__mod___features_conv4_14_c1x1_c_bn_running_mean, l__mod___features_conv4_14_c1x1_c_bn_running_var, l__mod___features_conv4_14_c1x1_c_bn_weight, l__mod___features_conv4_14_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_102 = l__mod___features_conv4_14_c1x1_c_bn_running_mean = l__mod___features_conv4_14_c1x1_c_bn_running_var = l__mod___features_conv4_14_c1x1_c_bn_weight = l__mod___features_conv4_14_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_246 = self.L__mod___features_conv4_14_c1x1_c_bn_drop(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_247 = self.L__mod___features_conv4_14_c1x1_c_bn_act(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_103 = self.L__mod___features_conv4_14_c1x1_c_conv(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_25 = x_in_103[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_25 = x_in_103[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_26 = x_s1_25 + out1_25;  x_s1_25 = out1_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_26 = torch.cat([x_s2_25, out2_25], dim = 1);  x_s2_25 = out2_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_104 = torch.cat((x_s1_26, x_s2_26), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_15_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_15_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__82 = l__mod___features_conv4_15_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_15_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c1x1_a_bn_running_mean = self.L__mod___features_conv4_15_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c1x1_a_bn_running_var = self.L__mod___features_conv4_15_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_15_c1x1_a_bn_weight = self.L__mod___features_conv4_15_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_15_c1x1_a_bn_bias = self.L__mod___features_conv4_15_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_248 = torch.nn.functional.batch_norm(x_in_104, l__mod___features_conv4_15_c1x1_a_bn_running_mean, l__mod___features_conv4_15_c1x1_a_bn_running_var, l__mod___features_conv4_15_c1x1_a_bn_weight, l__mod___features_conv4_15_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_104 = l__mod___features_conv4_15_c1x1_a_bn_running_mean = l__mod___features_conv4_15_c1x1_a_bn_running_var = l__mod___features_conv4_15_c1x1_a_bn_weight = l__mod___features_conv4_15_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_249 = self.L__mod___features_conv4_15_c1x1_a_bn_drop(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_250 = self.L__mod___features_conv4_15_c1x1_a_bn_act(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_105 = self.L__mod___features_conv4_15_c1x1_a_conv(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_15_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_15_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__83 = l__mod___features_conv4_15_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_15_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c3x3_b_bn_running_mean = self.L__mod___features_conv4_15_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c3x3_b_bn_running_var = self.L__mod___features_conv4_15_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_15_c3x3_b_bn_weight = self.L__mod___features_conv4_15_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_15_c3x3_b_bn_bias = self.L__mod___features_conv4_15_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_251 = torch.nn.functional.batch_norm(x_in_105, l__mod___features_conv4_15_c3x3_b_bn_running_mean, l__mod___features_conv4_15_c3x3_b_bn_running_var, l__mod___features_conv4_15_c3x3_b_bn_weight, l__mod___features_conv4_15_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_105 = l__mod___features_conv4_15_c3x3_b_bn_running_mean = l__mod___features_conv4_15_c3x3_b_bn_running_var = l__mod___features_conv4_15_c3x3_b_bn_weight = l__mod___features_conv4_15_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_252 = self.L__mod___features_conv4_15_c3x3_b_bn_drop(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_253 = self.L__mod___features_conv4_15_c3x3_b_bn_act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_106 = self.L__mod___features_conv4_15_c3x3_b_conv(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_15_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_15_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__84 = l__mod___features_conv4_15_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_15_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c1x1_c_bn_running_mean = self.L__mod___features_conv4_15_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_15_c1x1_c_bn_running_var = self.L__mod___features_conv4_15_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_15_c1x1_c_bn_weight = self.L__mod___features_conv4_15_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_15_c1x1_c_bn_bias = self.L__mod___features_conv4_15_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_254 = torch.nn.functional.batch_norm(x_in_106, l__mod___features_conv4_15_c1x1_c_bn_running_mean, l__mod___features_conv4_15_c1x1_c_bn_running_var, l__mod___features_conv4_15_c1x1_c_bn_weight, l__mod___features_conv4_15_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_106 = l__mod___features_conv4_15_c1x1_c_bn_running_mean = l__mod___features_conv4_15_c1x1_c_bn_running_var = l__mod___features_conv4_15_c1x1_c_bn_weight = l__mod___features_conv4_15_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_255 = self.L__mod___features_conv4_15_c1x1_c_bn_drop(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_256 = self.L__mod___features_conv4_15_c1x1_c_bn_act(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_107 = self.L__mod___features_conv4_15_c1x1_c_conv(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_26 = x_in_107[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_26 = x_in_107[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_27 = x_s1_26 + out1_26;  x_s1_26 = out1_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_27 = torch.cat([x_s2_26, out2_26], dim = 1);  x_s2_26 = out2_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_108 = torch.cat((x_s1_27, x_s2_27), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_16_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_16_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__85 = l__mod___features_conv4_16_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_16_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c1x1_a_bn_running_mean = self.L__mod___features_conv4_16_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c1x1_a_bn_running_var = self.L__mod___features_conv4_16_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_16_c1x1_a_bn_weight = self.L__mod___features_conv4_16_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_16_c1x1_a_bn_bias = self.L__mod___features_conv4_16_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_257 = torch.nn.functional.batch_norm(x_in_108, l__mod___features_conv4_16_c1x1_a_bn_running_mean, l__mod___features_conv4_16_c1x1_a_bn_running_var, l__mod___features_conv4_16_c1x1_a_bn_weight, l__mod___features_conv4_16_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_108 = l__mod___features_conv4_16_c1x1_a_bn_running_mean = l__mod___features_conv4_16_c1x1_a_bn_running_var = l__mod___features_conv4_16_c1x1_a_bn_weight = l__mod___features_conv4_16_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_258 = self.L__mod___features_conv4_16_c1x1_a_bn_drop(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_259 = self.L__mod___features_conv4_16_c1x1_a_bn_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_109 = self.L__mod___features_conv4_16_c1x1_a_conv(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_16_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_16_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__86 = l__mod___features_conv4_16_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_16_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c3x3_b_bn_running_mean = self.L__mod___features_conv4_16_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c3x3_b_bn_running_var = self.L__mod___features_conv4_16_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_16_c3x3_b_bn_weight = self.L__mod___features_conv4_16_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_16_c3x3_b_bn_bias = self.L__mod___features_conv4_16_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_260 = torch.nn.functional.batch_norm(x_in_109, l__mod___features_conv4_16_c3x3_b_bn_running_mean, l__mod___features_conv4_16_c3x3_b_bn_running_var, l__mod___features_conv4_16_c3x3_b_bn_weight, l__mod___features_conv4_16_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_109 = l__mod___features_conv4_16_c3x3_b_bn_running_mean = l__mod___features_conv4_16_c3x3_b_bn_running_var = l__mod___features_conv4_16_c3x3_b_bn_weight = l__mod___features_conv4_16_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_261 = self.L__mod___features_conv4_16_c3x3_b_bn_drop(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_262 = self.L__mod___features_conv4_16_c3x3_b_bn_act(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_110 = self.L__mod___features_conv4_16_c3x3_b_conv(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_16_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_16_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__87 = l__mod___features_conv4_16_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_16_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c1x1_c_bn_running_mean = self.L__mod___features_conv4_16_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_16_c1x1_c_bn_running_var = self.L__mod___features_conv4_16_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_16_c1x1_c_bn_weight = self.L__mod___features_conv4_16_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_16_c1x1_c_bn_bias = self.L__mod___features_conv4_16_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_263 = torch.nn.functional.batch_norm(x_in_110, l__mod___features_conv4_16_c1x1_c_bn_running_mean, l__mod___features_conv4_16_c1x1_c_bn_running_var, l__mod___features_conv4_16_c1x1_c_bn_weight, l__mod___features_conv4_16_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_110 = l__mod___features_conv4_16_c1x1_c_bn_running_mean = l__mod___features_conv4_16_c1x1_c_bn_running_var = l__mod___features_conv4_16_c1x1_c_bn_weight = l__mod___features_conv4_16_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_264 = self.L__mod___features_conv4_16_c1x1_c_bn_drop(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_265 = self.L__mod___features_conv4_16_c1x1_c_bn_act(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_111 = self.L__mod___features_conv4_16_c1x1_c_conv(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_27 = x_in_111[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_27 = x_in_111[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_28 = x_s1_27 + out1_27;  x_s1_27 = out1_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_28 = torch.cat([x_s2_27, out2_27], dim = 1);  x_s2_27 = out2_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_112 = torch.cat((x_s1_28, x_s2_28), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_17_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_17_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__88 = l__mod___features_conv4_17_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_17_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c1x1_a_bn_running_mean = self.L__mod___features_conv4_17_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c1x1_a_bn_running_var = self.L__mod___features_conv4_17_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_17_c1x1_a_bn_weight = self.L__mod___features_conv4_17_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_17_c1x1_a_bn_bias = self.L__mod___features_conv4_17_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_266 = torch.nn.functional.batch_norm(x_in_112, l__mod___features_conv4_17_c1x1_a_bn_running_mean, l__mod___features_conv4_17_c1x1_a_bn_running_var, l__mod___features_conv4_17_c1x1_a_bn_weight, l__mod___features_conv4_17_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_112 = l__mod___features_conv4_17_c1x1_a_bn_running_mean = l__mod___features_conv4_17_c1x1_a_bn_running_var = l__mod___features_conv4_17_c1x1_a_bn_weight = l__mod___features_conv4_17_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_267 = self.L__mod___features_conv4_17_c1x1_a_bn_drop(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_268 = self.L__mod___features_conv4_17_c1x1_a_bn_act(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_113 = self.L__mod___features_conv4_17_c1x1_a_conv(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_17_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_17_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__89 = l__mod___features_conv4_17_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_17_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c3x3_b_bn_running_mean = self.L__mod___features_conv4_17_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c3x3_b_bn_running_var = self.L__mod___features_conv4_17_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_17_c3x3_b_bn_weight = self.L__mod___features_conv4_17_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_17_c3x3_b_bn_bias = self.L__mod___features_conv4_17_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_269 = torch.nn.functional.batch_norm(x_in_113, l__mod___features_conv4_17_c3x3_b_bn_running_mean, l__mod___features_conv4_17_c3x3_b_bn_running_var, l__mod___features_conv4_17_c3x3_b_bn_weight, l__mod___features_conv4_17_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_113 = l__mod___features_conv4_17_c3x3_b_bn_running_mean = l__mod___features_conv4_17_c3x3_b_bn_running_var = l__mod___features_conv4_17_c3x3_b_bn_weight = l__mod___features_conv4_17_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_270 = self.L__mod___features_conv4_17_c3x3_b_bn_drop(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_271 = self.L__mod___features_conv4_17_c3x3_b_bn_act(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_114 = self.L__mod___features_conv4_17_c3x3_b_conv(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_17_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_17_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__90 = l__mod___features_conv4_17_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_17_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c1x1_c_bn_running_mean = self.L__mod___features_conv4_17_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_17_c1x1_c_bn_running_var = self.L__mod___features_conv4_17_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_17_c1x1_c_bn_weight = self.L__mod___features_conv4_17_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_17_c1x1_c_bn_bias = self.L__mod___features_conv4_17_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_272 = torch.nn.functional.batch_norm(x_in_114, l__mod___features_conv4_17_c1x1_c_bn_running_mean, l__mod___features_conv4_17_c1x1_c_bn_running_var, l__mod___features_conv4_17_c1x1_c_bn_weight, l__mod___features_conv4_17_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_114 = l__mod___features_conv4_17_c1x1_c_bn_running_mean = l__mod___features_conv4_17_c1x1_c_bn_running_var = l__mod___features_conv4_17_c1x1_c_bn_weight = l__mod___features_conv4_17_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_273 = self.L__mod___features_conv4_17_c1x1_c_bn_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_274 = self.L__mod___features_conv4_17_c1x1_c_bn_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_115 = self.L__mod___features_conv4_17_c1x1_c_conv(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_28 = x_in_115[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_28 = x_in_115[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_29 = x_s1_28 + out1_28;  x_s1_28 = out1_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_29 = torch.cat([x_s2_28, out2_28], dim = 1);  x_s2_28 = out2_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_116 = torch.cat((x_s1_29, x_s2_29), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_18_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_18_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__91 = l__mod___features_conv4_18_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_18_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c1x1_a_bn_running_mean = self.L__mod___features_conv4_18_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c1x1_a_bn_running_var = self.L__mod___features_conv4_18_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_18_c1x1_a_bn_weight = self.L__mod___features_conv4_18_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_18_c1x1_a_bn_bias = self.L__mod___features_conv4_18_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_275 = torch.nn.functional.batch_norm(x_in_116, l__mod___features_conv4_18_c1x1_a_bn_running_mean, l__mod___features_conv4_18_c1x1_a_bn_running_var, l__mod___features_conv4_18_c1x1_a_bn_weight, l__mod___features_conv4_18_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_116 = l__mod___features_conv4_18_c1x1_a_bn_running_mean = l__mod___features_conv4_18_c1x1_a_bn_running_var = l__mod___features_conv4_18_c1x1_a_bn_weight = l__mod___features_conv4_18_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_276 = self.L__mod___features_conv4_18_c1x1_a_bn_drop(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_277 = self.L__mod___features_conv4_18_c1x1_a_bn_act(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_117 = self.L__mod___features_conv4_18_c1x1_a_conv(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_18_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_18_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__92 = l__mod___features_conv4_18_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_18_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c3x3_b_bn_running_mean = self.L__mod___features_conv4_18_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c3x3_b_bn_running_var = self.L__mod___features_conv4_18_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_18_c3x3_b_bn_weight = self.L__mod___features_conv4_18_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_18_c3x3_b_bn_bias = self.L__mod___features_conv4_18_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_278 = torch.nn.functional.batch_norm(x_in_117, l__mod___features_conv4_18_c3x3_b_bn_running_mean, l__mod___features_conv4_18_c3x3_b_bn_running_var, l__mod___features_conv4_18_c3x3_b_bn_weight, l__mod___features_conv4_18_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_117 = l__mod___features_conv4_18_c3x3_b_bn_running_mean = l__mod___features_conv4_18_c3x3_b_bn_running_var = l__mod___features_conv4_18_c3x3_b_bn_weight = l__mod___features_conv4_18_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_279 = self.L__mod___features_conv4_18_c3x3_b_bn_drop(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_280 = self.L__mod___features_conv4_18_c3x3_b_bn_act(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_118 = self.L__mod___features_conv4_18_c3x3_b_conv(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_18_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_18_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__93 = l__mod___features_conv4_18_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_18_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c1x1_c_bn_running_mean = self.L__mod___features_conv4_18_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_18_c1x1_c_bn_running_var = self.L__mod___features_conv4_18_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_18_c1x1_c_bn_weight = self.L__mod___features_conv4_18_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_18_c1x1_c_bn_bias = self.L__mod___features_conv4_18_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_281 = torch.nn.functional.batch_norm(x_in_118, l__mod___features_conv4_18_c1x1_c_bn_running_mean, l__mod___features_conv4_18_c1x1_c_bn_running_var, l__mod___features_conv4_18_c1x1_c_bn_weight, l__mod___features_conv4_18_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_118 = l__mod___features_conv4_18_c1x1_c_bn_running_mean = l__mod___features_conv4_18_c1x1_c_bn_running_var = l__mod___features_conv4_18_c1x1_c_bn_weight = l__mod___features_conv4_18_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_282 = self.L__mod___features_conv4_18_c1x1_c_bn_drop(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_283 = self.L__mod___features_conv4_18_c1x1_c_bn_act(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_119 = self.L__mod___features_conv4_18_c1x1_c_conv(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_29 = x_in_119[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_29 = x_in_119[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_30 = x_s1_29 + out1_29;  x_s1_29 = out1_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_30 = torch.cat([x_s2_29, out2_29], dim = 1);  x_s2_29 = out2_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_120 = torch.cat((x_s1_30, x_s2_30), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_19_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_19_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__94 = l__mod___features_conv4_19_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_19_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c1x1_a_bn_running_mean = self.L__mod___features_conv4_19_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c1x1_a_bn_running_var = self.L__mod___features_conv4_19_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_19_c1x1_a_bn_weight = self.L__mod___features_conv4_19_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_19_c1x1_a_bn_bias = self.L__mod___features_conv4_19_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_284 = torch.nn.functional.batch_norm(x_in_120, l__mod___features_conv4_19_c1x1_a_bn_running_mean, l__mod___features_conv4_19_c1x1_a_bn_running_var, l__mod___features_conv4_19_c1x1_a_bn_weight, l__mod___features_conv4_19_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_120 = l__mod___features_conv4_19_c1x1_a_bn_running_mean = l__mod___features_conv4_19_c1x1_a_bn_running_var = l__mod___features_conv4_19_c1x1_a_bn_weight = l__mod___features_conv4_19_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_285 = self.L__mod___features_conv4_19_c1x1_a_bn_drop(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_286 = self.L__mod___features_conv4_19_c1x1_a_bn_act(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_121 = self.L__mod___features_conv4_19_c1x1_a_conv(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_19_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_19_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__95 = l__mod___features_conv4_19_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_19_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c3x3_b_bn_running_mean = self.L__mod___features_conv4_19_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c3x3_b_bn_running_var = self.L__mod___features_conv4_19_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_19_c3x3_b_bn_weight = self.L__mod___features_conv4_19_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_19_c3x3_b_bn_bias = self.L__mod___features_conv4_19_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_287 = torch.nn.functional.batch_norm(x_in_121, l__mod___features_conv4_19_c3x3_b_bn_running_mean, l__mod___features_conv4_19_c3x3_b_bn_running_var, l__mod___features_conv4_19_c3x3_b_bn_weight, l__mod___features_conv4_19_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_121 = l__mod___features_conv4_19_c3x3_b_bn_running_mean = l__mod___features_conv4_19_c3x3_b_bn_running_var = l__mod___features_conv4_19_c3x3_b_bn_weight = l__mod___features_conv4_19_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_288 = self.L__mod___features_conv4_19_c3x3_b_bn_drop(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_289 = self.L__mod___features_conv4_19_c3x3_b_bn_act(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_122 = self.L__mod___features_conv4_19_c3x3_b_conv(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_19_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_19_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__96 = l__mod___features_conv4_19_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_19_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c1x1_c_bn_running_mean = self.L__mod___features_conv4_19_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_19_c1x1_c_bn_running_var = self.L__mod___features_conv4_19_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_19_c1x1_c_bn_weight = self.L__mod___features_conv4_19_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_19_c1x1_c_bn_bias = self.L__mod___features_conv4_19_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_290 = torch.nn.functional.batch_norm(x_in_122, l__mod___features_conv4_19_c1x1_c_bn_running_mean, l__mod___features_conv4_19_c1x1_c_bn_running_var, l__mod___features_conv4_19_c1x1_c_bn_weight, l__mod___features_conv4_19_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_122 = l__mod___features_conv4_19_c1x1_c_bn_running_mean = l__mod___features_conv4_19_c1x1_c_bn_running_var = l__mod___features_conv4_19_c1x1_c_bn_weight = l__mod___features_conv4_19_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_291 = self.L__mod___features_conv4_19_c1x1_c_bn_drop(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_292 = self.L__mod___features_conv4_19_c1x1_c_bn_act(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_123 = self.L__mod___features_conv4_19_c1x1_c_conv(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_30 = x_in_123[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_30 = x_in_123[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_31 = x_s1_30 + out1_30;  x_s1_30 = out1_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_31 = torch.cat([x_s2_30, out2_30], dim = 1);  x_s2_30 = out2_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_124 = torch.cat((x_s1_31, x_s2_31), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_20_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv4_20_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__97 = l__mod___features_conv4_20_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_20_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c1x1_a_bn_running_mean = self.L__mod___features_conv4_20_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c1x1_a_bn_running_var = self.L__mod___features_conv4_20_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_20_c1x1_a_bn_weight = self.L__mod___features_conv4_20_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_20_c1x1_a_bn_bias = self.L__mod___features_conv4_20_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_293 = torch.nn.functional.batch_norm(x_in_124, l__mod___features_conv4_20_c1x1_a_bn_running_mean, l__mod___features_conv4_20_c1x1_a_bn_running_var, l__mod___features_conv4_20_c1x1_a_bn_weight, l__mod___features_conv4_20_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_124 = l__mod___features_conv4_20_c1x1_a_bn_running_mean = l__mod___features_conv4_20_c1x1_a_bn_running_var = l__mod___features_conv4_20_c1x1_a_bn_weight = l__mod___features_conv4_20_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_294 = self.L__mod___features_conv4_20_c1x1_a_bn_drop(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_295 = self.L__mod___features_conv4_20_c1x1_a_bn_act(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_125 = self.L__mod___features_conv4_20_c1x1_a_conv(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_20_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv4_20_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__98 = l__mod___features_conv4_20_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_20_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c3x3_b_bn_running_mean = self.L__mod___features_conv4_20_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c3x3_b_bn_running_var = self.L__mod___features_conv4_20_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_20_c3x3_b_bn_weight = self.L__mod___features_conv4_20_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_20_c3x3_b_bn_bias = self.L__mod___features_conv4_20_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_296 = torch.nn.functional.batch_norm(x_in_125, l__mod___features_conv4_20_c3x3_b_bn_running_mean, l__mod___features_conv4_20_c3x3_b_bn_running_var, l__mod___features_conv4_20_c3x3_b_bn_weight, l__mod___features_conv4_20_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_125 = l__mod___features_conv4_20_c3x3_b_bn_running_mean = l__mod___features_conv4_20_c3x3_b_bn_running_var = l__mod___features_conv4_20_c3x3_b_bn_weight = l__mod___features_conv4_20_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_297 = self.L__mod___features_conv4_20_c3x3_b_bn_drop(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_298 = self.L__mod___features_conv4_20_c3x3_b_bn_act(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_126 = self.L__mod___features_conv4_20_c3x3_b_conv(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv4_20_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv4_20_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__99 = l__mod___features_conv4_20_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv4_20_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c1x1_c_bn_running_mean = self.L__mod___features_conv4_20_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv4_20_c1x1_c_bn_running_var = self.L__mod___features_conv4_20_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv4_20_c1x1_c_bn_weight = self.L__mod___features_conv4_20_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv4_20_c1x1_c_bn_bias = self.L__mod___features_conv4_20_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_299 = torch.nn.functional.batch_norm(x_in_126, l__mod___features_conv4_20_c1x1_c_bn_running_mean, l__mod___features_conv4_20_c1x1_c_bn_running_var, l__mod___features_conv4_20_c1x1_c_bn_weight, l__mod___features_conv4_20_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_126 = l__mod___features_conv4_20_c1x1_c_bn_running_mean = l__mod___features_conv4_20_c1x1_c_bn_running_var = l__mod___features_conv4_20_c1x1_c_bn_weight = l__mod___features_conv4_20_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_300 = self.L__mod___features_conv4_20_c1x1_c_bn_drop(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_301 = self.L__mod___features_conv4_20_c1x1_c_bn_act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_127 = self.L__mod___features_conv4_20_c1x1_c_conv(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_31 = x_in_127[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_31 = x_in_127[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  x_in_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    resid_31 = x_s1_31 + out1_31;  x_s1_31 = out1_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    dense_31 = torch.cat([x_s2_31, out2_31], dim = 1);  x_s2_31 = out2_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_128 = torch.cat((resid_31, dense_31), dim = 1);  resid_31 = dense_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_1_c1x1_w_s2_bn_num_batches_tracked = self.L__mod___features_conv5_1_c1x1_w_s2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__100 = l__mod___features_conv5_1_c1x1_w_s2_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_1_c1x1_w_s2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_w_s2_bn_running_mean = self.L__mod___features_conv5_1_c1x1_w_s2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_w_s2_bn_running_var = self.L__mod___features_conv5_1_c1x1_w_s2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_1_c1x1_w_s2_bn_weight = self.L__mod___features_conv5_1_c1x1_w_s2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_1_c1x1_w_s2_bn_bias = self.L__mod___features_conv5_1_c1x1_w_s2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_302 = torch.nn.functional.batch_norm(x_in_128, l__mod___features_conv5_1_c1x1_w_s2_bn_running_mean, l__mod___features_conv5_1_c1x1_w_s2_bn_running_var, l__mod___features_conv5_1_c1x1_w_s2_bn_weight, l__mod___features_conv5_1_c1x1_w_s2_bn_bias, True, 0.1, 0.001);  l__mod___features_conv5_1_c1x1_w_s2_bn_running_mean = l__mod___features_conv5_1_c1x1_w_s2_bn_running_var = l__mod___features_conv5_1_c1x1_w_s2_bn_weight = l__mod___features_conv5_1_c1x1_w_s2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_303 = self.L__mod___features_conv5_1_c1x1_w_s2_bn_drop(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_304 = self.L__mod___features_conv5_1_c1x1_w_s2_bn_act(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_s_3 = self.L__mod___features_conv5_1_c1x1_w_s2_conv(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    x_s1_32 = x_s_3[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    x_s2_32 = x_s_3[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))];  x_s_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_1_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv5_1_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__101 = l__mod___features_conv5_1_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_1_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_a_bn_running_mean = self.L__mod___features_conv5_1_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_a_bn_running_var = self.L__mod___features_conv5_1_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_1_c1x1_a_bn_weight = self.L__mod___features_conv5_1_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_1_c1x1_a_bn_bias = self.L__mod___features_conv5_1_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_305 = torch.nn.functional.batch_norm(x_in_128, l__mod___features_conv5_1_c1x1_a_bn_running_mean, l__mod___features_conv5_1_c1x1_a_bn_running_var, l__mod___features_conv5_1_c1x1_a_bn_weight, l__mod___features_conv5_1_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_128 = l__mod___features_conv5_1_c1x1_a_bn_running_mean = l__mod___features_conv5_1_c1x1_a_bn_running_var = l__mod___features_conv5_1_c1x1_a_bn_weight = l__mod___features_conv5_1_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_306 = self.L__mod___features_conv5_1_c1x1_a_bn_drop(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_307 = self.L__mod___features_conv5_1_c1x1_a_bn_act(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_129 = self.L__mod___features_conv5_1_c1x1_a_conv(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_1_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv5_1_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__102 = l__mod___features_conv5_1_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_1_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c3x3_b_bn_running_mean = self.L__mod___features_conv5_1_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c3x3_b_bn_running_var = self.L__mod___features_conv5_1_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_1_c3x3_b_bn_weight = self.L__mod___features_conv5_1_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_1_c3x3_b_bn_bias = self.L__mod___features_conv5_1_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_308 = torch.nn.functional.batch_norm(x_in_129, l__mod___features_conv5_1_c3x3_b_bn_running_mean, l__mod___features_conv5_1_c3x3_b_bn_running_var, l__mod___features_conv5_1_c3x3_b_bn_weight, l__mod___features_conv5_1_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_129 = l__mod___features_conv5_1_c3x3_b_bn_running_mean = l__mod___features_conv5_1_c3x3_b_bn_running_var = l__mod___features_conv5_1_c3x3_b_bn_weight = l__mod___features_conv5_1_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_309 = self.L__mod___features_conv5_1_c3x3_b_bn_drop(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_310 = self.L__mod___features_conv5_1_c3x3_b_bn_act(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_130 = self.L__mod___features_conv5_1_c3x3_b_conv(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_1_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv5_1_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__103 = l__mod___features_conv5_1_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_1_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_c_bn_running_mean = self.L__mod___features_conv5_1_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_1_c1x1_c_bn_running_var = self.L__mod___features_conv5_1_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_1_c1x1_c_bn_weight = self.L__mod___features_conv5_1_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_1_c1x1_c_bn_bias = self.L__mod___features_conv5_1_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_311 = torch.nn.functional.batch_norm(x_in_130, l__mod___features_conv5_1_c1x1_c_bn_running_mean, l__mod___features_conv5_1_c1x1_c_bn_running_var, l__mod___features_conv5_1_c1x1_c_bn_weight, l__mod___features_conv5_1_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_130 = l__mod___features_conv5_1_c1x1_c_bn_running_mean = l__mod___features_conv5_1_c1x1_c_bn_running_var = l__mod___features_conv5_1_c1x1_c_bn_weight = l__mod___features_conv5_1_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_312 = self.L__mod___features_conv5_1_c1x1_c_bn_drop(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_313 = self.L__mod___features_conv5_1_c1x1_c_bn_act(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_131 = self.L__mod___features_conv5_1_c1x1_c_conv(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_32 = x_in_131[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_32 = x_in_131[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))];  x_in_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_33 = x_s1_32 + out1_32;  x_s1_32 = out1_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_33 = torch.cat([x_s2_32, out2_32], dim = 1);  x_s2_32 = out2_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_132 = torch.cat((x_s1_33, x_s2_33), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_2_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv5_2_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__104 = l__mod___features_conv5_2_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_2_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c1x1_a_bn_running_mean = self.L__mod___features_conv5_2_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c1x1_a_bn_running_var = self.L__mod___features_conv5_2_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_2_c1x1_a_bn_weight = self.L__mod___features_conv5_2_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_2_c1x1_a_bn_bias = self.L__mod___features_conv5_2_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_314 = torch.nn.functional.batch_norm(x_in_132, l__mod___features_conv5_2_c1x1_a_bn_running_mean, l__mod___features_conv5_2_c1x1_a_bn_running_var, l__mod___features_conv5_2_c1x1_a_bn_weight, l__mod___features_conv5_2_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_132 = l__mod___features_conv5_2_c1x1_a_bn_running_mean = l__mod___features_conv5_2_c1x1_a_bn_running_var = l__mod___features_conv5_2_c1x1_a_bn_weight = l__mod___features_conv5_2_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_315 = self.L__mod___features_conv5_2_c1x1_a_bn_drop(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_316 = self.L__mod___features_conv5_2_c1x1_a_bn_act(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_133 = self.L__mod___features_conv5_2_c1x1_a_conv(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_2_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv5_2_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__105 = l__mod___features_conv5_2_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_2_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c3x3_b_bn_running_mean = self.L__mod___features_conv5_2_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c3x3_b_bn_running_var = self.L__mod___features_conv5_2_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_2_c3x3_b_bn_weight = self.L__mod___features_conv5_2_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_2_c3x3_b_bn_bias = self.L__mod___features_conv5_2_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_317 = torch.nn.functional.batch_norm(x_in_133, l__mod___features_conv5_2_c3x3_b_bn_running_mean, l__mod___features_conv5_2_c3x3_b_bn_running_var, l__mod___features_conv5_2_c3x3_b_bn_weight, l__mod___features_conv5_2_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_133 = l__mod___features_conv5_2_c3x3_b_bn_running_mean = l__mod___features_conv5_2_c3x3_b_bn_running_var = l__mod___features_conv5_2_c3x3_b_bn_weight = l__mod___features_conv5_2_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_318 = self.L__mod___features_conv5_2_c3x3_b_bn_drop(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_319 = self.L__mod___features_conv5_2_c3x3_b_bn_act(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_134 = self.L__mod___features_conv5_2_c3x3_b_conv(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_2_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv5_2_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__106 = l__mod___features_conv5_2_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_2_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c1x1_c_bn_running_mean = self.L__mod___features_conv5_2_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_2_c1x1_c_bn_running_var = self.L__mod___features_conv5_2_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_2_c1x1_c_bn_weight = self.L__mod___features_conv5_2_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_2_c1x1_c_bn_bias = self.L__mod___features_conv5_2_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_320 = torch.nn.functional.batch_norm(x_in_134, l__mod___features_conv5_2_c1x1_c_bn_running_mean, l__mod___features_conv5_2_c1x1_c_bn_running_var, l__mod___features_conv5_2_c1x1_c_bn_weight, l__mod___features_conv5_2_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_134 = l__mod___features_conv5_2_c1x1_c_bn_running_mean = l__mod___features_conv5_2_c1x1_c_bn_running_var = l__mod___features_conv5_2_c1x1_c_bn_weight = l__mod___features_conv5_2_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_321 = self.L__mod___features_conv5_2_c1x1_c_bn_drop(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_322 = self.L__mod___features_conv5_2_c1x1_c_bn_act(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_135 = self.L__mod___features_conv5_2_c1x1_c_conv(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_33 = x_in_135[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_33 = x_in_135[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))];  x_in_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    x_s1_34 = x_s1_33 + out1_33;  x_s1_33 = out1_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    x_s2_34 = torch.cat([x_s2_33, out2_33], dim = 1);  x_s2_33 = out2_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    x_in_136 = torch.cat((x_s1_34, x_s2_34), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_3_c1x1_a_bn_num_batches_tracked = self.L__mod___features_conv5_3_c1x1_a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__107 = l__mod___features_conv5_3_c1x1_a_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_3_c1x1_a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c1x1_a_bn_running_mean = self.L__mod___features_conv5_3_c1x1_a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c1x1_a_bn_running_var = self.L__mod___features_conv5_3_c1x1_a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_3_c1x1_a_bn_weight = self.L__mod___features_conv5_3_c1x1_a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_3_c1x1_a_bn_bias = self.L__mod___features_conv5_3_c1x1_a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_323 = torch.nn.functional.batch_norm(x_in_136, l__mod___features_conv5_3_c1x1_a_bn_running_mean, l__mod___features_conv5_3_c1x1_a_bn_running_var, l__mod___features_conv5_3_c1x1_a_bn_weight, l__mod___features_conv5_3_c1x1_a_bn_bias, True, 0.1, 0.001);  x_in_136 = l__mod___features_conv5_3_c1x1_a_bn_running_mean = l__mod___features_conv5_3_c1x1_a_bn_running_var = l__mod___features_conv5_3_c1x1_a_bn_weight = l__mod___features_conv5_3_c1x1_a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_324 = self.L__mod___features_conv5_3_c1x1_a_bn_drop(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_325 = self.L__mod___features_conv5_3_c1x1_a_bn_act(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_137 = self.L__mod___features_conv5_3_c1x1_a_conv(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_3_c3x3_b_bn_num_batches_tracked = self.L__mod___features_conv5_3_c3x3_b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__108 = l__mod___features_conv5_3_c3x3_b_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_3_c3x3_b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c3x3_b_bn_running_mean = self.L__mod___features_conv5_3_c3x3_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c3x3_b_bn_running_var = self.L__mod___features_conv5_3_c3x3_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_3_c3x3_b_bn_weight = self.L__mod___features_conv5_3_c3x3_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_3_c3x3_b_bn_bias = self.L__mod___features_conv5_3_c3x3_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_326 = torch.nn.functional.batch_norm(x_in_137, l__mod___features_conv5_3_c3x3_b_bn_running_mean, l__mod___features_conv5_3_c3x3_b_bn_running_var, l__mod___features_conv5_3_c3x3_b_bn_weight, l__mod___features_conv5_3_c3x3_b_bn_bias, True, 0.1, 0.001);  x_in_137 = l__mod___features_conv5_3_c3x3_b_bn_running_mean = l__mod___features_conv5_3_c3x3_b_bn_running_var = l__mod___features_conv5_3_c3x3_b_bn_weight = l__mod___features_conv5_3_c3x3_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_327 = self.L__mod___features_conv5_3_c3x3_b_bn_drop(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_328 = self.L__mod___features_conv5_3_c3x3_b_bn_act(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_138 = self.L__mod___features_conv5_3_c3x3_b_conv(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_3_c1x1_c_bn_num_batches_tracked = self.L__mod___features_conv5_3_c1x1_c_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__109 = l__mod___features_conv5_3_c1x1_c_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_3_c1x1_c_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c1x1_c_bn_running_mean = self.L__mod___features_conv5_3_c1x1_c_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_3_c1x1_c_bn_running_var = self.L__mod___features_conv5_3_c1x1_c_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_3_c1x1_c_bn_weight = self.L__mod___features_conv5_3_c1x1_c_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_3_c1x1_c_bn_bias = self.L__mod___features_conv5_3_c1x1_c_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_329 = torch.nn.functional.batch_norm(x_in_138, l__mod___features_conv5_3_c1x1_c_bn_running_mean, l__mod___features_conv5_3_c1x1_c_bn_running_var, l__mod___features_conv5_3_c1x1_c_bn_weight, l__mod___features_conv5_3_c1x1_c_bn_bias, True, 0.1, 0.001);  x_in_138 = l__mod___features_conv5_3_c1x1_c_bn_running_mean = l__mod___features_conv5_3_c1x1_c_bn_running_var = l__mod___features_conv5_3_c1x1_c_bn_weight = l__mod___features_conv5_3_c1x1_c_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_330 = self.L__mod___features_conv5_3_c1x1_c_bn_drop(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_331 = self.L__mod___features_conv5_3_c1x1_c_bn_act(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    x_in_139 = self.L__mod___features_conv5_3_c1x1_c_conv(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    out1_34 = x_in_139[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    out2_34 = x_in_139[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))];  x_in_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    resid_34 = x_s1_34 + out1_34;  x_s1_34 = out1_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    dense_34 = torch.cat([x_s2_34, out2_34], dim = 1);  x_s2_34 = out2_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:42, code: x = torch.cat(x, dim=1)
    x_332 = torch.cat((resid_34, dense_34), dim = 1);  resid_34 = dense_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___features_conv5_bn_ac_bn_num_batches_tracked = self.L__mod___features_conv5_bn_ac_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__110 = l__mod___features_conv5_bn_ac_bn_num_batches_tracked.add_(1);  l__mod___features_conv5_bn_ac_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_bn_ac_bn_running_mean = self.L__mod___features_conv5_bn_ac_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___features_conv5_bn_ac_bn_running_var = self.L__mod___features_conv5_bn_ac_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___features_conv5_bn_ac_bn_weight = self.L__mod___features_conv5_bn_ac_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___features_conv5_bn_ac_bn_bias = self.L__mod___features_conv5_bn_ac_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_333 = torch.nn.functional.batch_norm(x_332, l__mod___features_conv5_bn_ac_bn_running_mean, l__mod___features_conv5_bn_ac_bn_running_var, l__mod___features_conv5_bn_ac_bn_weight, l__mod___features_conv5_bn_ac_bn_bias, True, 0.1, 0.001);  x_332 = l__mod___features_conv5_bn_ac_bn_running_mean = l__mod___features_conv5_bn_ac_bn_running_var = l__mod___features_conv5_bn_ac_bn_weight = l__mod___features_conv5_bn_ac_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_334 = self.L__mod___features_conv5_bn_ac_bn_drop(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_336 = self.L__mod___features_conv5_bn_ac_bn_act(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_337 = self.L__mod___global_pool_pool(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_339 = self.L__mod___global_pool_flatten(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:274, code: x = self.classifier(x)
    x_340 = self.L__mod___classifier(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:275, code: return self.flatten(x)
    pred = self.L__mod___flatten(x_340);  x_340 = None
    return (pred,)
    