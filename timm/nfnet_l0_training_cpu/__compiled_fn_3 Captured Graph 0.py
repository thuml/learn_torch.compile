from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    l__mod___stem_conv1_weight = self.L__mod___stem_conv1_weight
    reshape = l__mod___stem_conv1_weight.reshape(1, 16, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    l__mod___stem_conv1_gain = self.L__mod___stem_conv1_gain
    mul = l__mod___stem_conv1_gain * 0.34412564994580647;  l__mod___stem_conv1_gain = None
    view = mul.view(-1);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm = torch.nn.functional.batch_norm(reshape, None, None, weight = view, training = True, momentum = 0.0, eps = 1e-05);  reshape = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight = batch_norm.reshape_as(l__mod___stem_conv1_weight);  batch_norm = l__mod___stem_conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    l__mod___stem_conv1_bias = self.L__mod___stem_conv1_bias
    conv2d = torch.conv2d(l_cloned_inputs_0_, weight, l__mod___stem_conv1_bias, (2, 2), (1, 1), (1, 1), 1);  l_cloned_inputs_0_ = weight = l__mod___stem_conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    l__mod___stem_act2 = self.L__mod___stem_act2(conv2d);  conv2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    l__mod___stem_conv2_weight = self.L__mod___stem_conv2_weight
    reshape_1 = l__mod___stem_conv2_weight.reshape(1, 32, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    l__mod___stem_conv2_gain = self.L__mod___stem_conv2_gain
    mul_1 = l__mod___stem_conv2_gain * 0.1490107774734497;  l__mod___stem_conv2_gain = None
    view_1 = mul_1.view(-1);  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_1 = torch.nn.functional.batch_norm(reshape_1, None, None, weight = view_1, training = True, momentum = 0.0, eps = 1e-05);  reshape_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_1 = batch_norm_1.reshape_as(l__mod___stem_conv2_weight);  batch_norm_1 = l__mod___stem_conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    l__mod___stem_conv2_bias = self.L__mod___stem_conv2_bias
    conv2d_1 = torch.conv2d(l__mod___stem_act2, weight_1, l__mod___stem_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  l__mod___stem_act2 = weight_1 = l__mod___stem_conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    l__mod___stem_act3 = self.L__mod___stem_act3(conv2d_1);  conv2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    l__mod___stem_conv3_weight = self.L__mod___stem_conv3_weight
    reshape_2 = l__mod___stem_conv3_weight.reshape(1, 64, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    l__mod___stem_conv3_gain = self.L__mod___stem_conv3_gain
    mul_2 = l__mod___stem_conv3_gain * 0.10536653122135592;  l__mod___stem_conv3_gain = None
    view_2 = mul_2.view(-1);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_2 = torch.nn.functional.batch_norm(reshape_2, None, None, weight = view_2, training = True, momentum = 0.0, eps = 1e-05);  reshape_2 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_2 = batch_norm_2.reshape_as(l__mod___stem_conv3_weight);  batch_norm_2 = l__mod___stem_conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    l__mod___stem_conv3_bias = self.L__mod___stem_conv3_bias
    conv2d_2 = torch.conv2d(l__mod___stem_act3, weight_2, l__mod___stem_conv3_bias, (1, 1), (1, 1), (1, 1), 1);  l__mod___stem_act3 = weight_2 = l__mod___stem_conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    l__mod___stem_act4 = self.L__mod___stem_act4(conv2d_2);  conv2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    l__mod___stem_conv4_weight = self.L__mod___stem_conv4_weight
    reshape_3 = l__mod___stem_conv4_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    l__mod___stem_conv4_gain = self.L__mod___stem_conv4_gain
    mul_3 = l__mod___stem_conv4_gain * 0.07450538873672485;  l__mod___stem_conv4_gain = None
    view_3 = mul_3.view(-1);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_3 = torch.nn.functional.batch_norm(reshape_3, None, None, weight = view_3, training = True, momentum = 0.0, eps = 1e-05);  reshape_3 = view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_3 = batch_norm_3.reshape_as(l__mod___stem_conv4_weight);  batch_norm_3 = l__mod___stem_conv4_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    l__mod___stem_conv4_bias = self.L__mod___stem_conv4_bias
    shortcut = torch.conv2d(l__mod___stem_act4, weight_3, l__mod___stem_conv4_bias, (2, 2), (1, 1), (1, 1), 1);  l__mod___stem_act4 = weight_3 = l__mod___stem_conv4_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___0_____0___act1 = self.getattr_getattr_L__mod___stages___0_____0___act1(shortcut);  shortcut = None
    out = getattr_getattr_l__mod___stages___0_____0___act1 * 1.0;  getattr_getattr_l__mod___stages___0_____0___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    getattr_getattr_l__mod___stages___0_____0___downsample_pool = self.getattr_getattr_L__mod___stages___0_____0___downsample_pool(out)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___0_____0___downsample_conv_weight = self.getattr_getattr_L__mod___stages___0_____0___downsample_conv_weight
    reshape_4 = getattr_getattr_l__mod___stages___0_____0___downsample_conv_weight.reshape(1, 256, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___0_____0___downsample_conv_gain = self.getattr_getattr_L__mod___stages___0_____0___downsample_conv_gain
    mul_5 = getattr_getattr_l__mod___stages___0_____0___downsample_conv_gain * 0.1580497968320339;  getattr_getattr_l__mod___stages___0_____0___downsample_conv_gain = None
    view_4 = mul_5.view(-1);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_4 = torch.nn.functional.batch_norm(reshape_4, None, None, weight = view_4, training = True, momentum = 0.0, eps = 1e-05);  reshape_4 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_4 = batch_norm_4.reshape_as(getattr_getattr_l__mod___stages___0_____0___downsample_conv_weight);  batch_norm_4 = getattr_getattr_l__mod___stages___0_____0___downsample_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___0_____0___downsample_conv_bias = self.getattr_getattr_L__mod___stages___0_____0___downsample_conv_bias
    shortcut_1 = torch.conv2d(getattr_getattr_l__mod___stages___0_____0___downsample_pool, weight_4, getattr_getattr_l__mod___stages___0_____0___downsample_conv_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___0_____0___downsample_pool = weight_4 = getattr_getattr_l__mod___stages___0_____0___downsample_conv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___0_____0___conv1_weight = self.getattr_getattr_L__mod___stages___0_____0___conv1_weight
    reshape_5 = getattr_getattr_l__mod___stages___0_____0___conv1_weight.reshape(1, 64, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___0_____0___conv1_gain = self.getattr_getattr_L__mod___stages___0_____0___conv1_gain
    mul_6 = getattr_getattr_l__mod___stages___0_____0___conv1_gain * 0.1580497968320339;  getattr_getattr_l__mod___stages___0_____0___conv1_gain = None
    view_5 = mul_6.view(-1);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_5 = torch.nn.functional.batch_norm(reshape_5, None, None, weight = view_5, training = True, momentum = 0.0, eps = 1e-05);  reshape_5 = view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_5 = batch_norm_5.reshape_as(getattr_getattr_l__mod___stages___0_____0___conv1_weight);  batch_norm_5 = getattr_getattr_l__mod___stages___0_____0___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___0_____0___conv1_bias = self.getattr_getattr_L__mod___stages___0_____0___conv1_bias
    out_1 = torch.conv2d(out, weight_5, getattr_getattr_l__mod___stages___0_____0___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out = weight_5 = getattr_getattr_l__mod___stages___0_____0___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___0_____0___act2 = self.getattr_getattr_L__mod___stages___0_____0___act2(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___0_____0___conv2_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2_weight
    reshape_6 = getattr_getattr_l__mod___stages___0_____0___conv2_weight.reshape(1, 64, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___0_____0___conv2_gain = self.getattr_getattr_L__mod___stages___0_____0___conv2_gain
    mul_7 = getattr_getattr_l__mod___stages___0_____0___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___0_____0___conv2_gain = None
    view_6 = mul_7.view(-1);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_6 = torch.nn.functional.batch_norm(reshape_6, None, None, weight = view_6, training = True, momentum = 0.0, eps = 1e-05);  reshape_6 = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_6 = batch_norm_6.reshape_as(getattr_getattr_l__mod___stages___0_____0___conv2_weight);  batch_norm_6 = getattr_getattr_l__mod___stages___0_____0___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___0_____0___conv2_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2_bias
    out_2 = torch.conv2d(getattr_getattr_l__mod___stages___0_____0___act2, weight_6, getattr_getattr_l__mod___stages___0_____0___conv2_bias, (1, 1), (1, 1), (1, 1), 1);  getattr_getattr_l__mod___stages___0_____0___act2 = weight_6 = getattr_getattr_l__mod___stages___0_____0___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___0_____0___act2b = self.getattr_getattr_L__mod___stages___0_____0___act2b(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___0_____0___conv2b_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2b_weight
    reshape_7 = getattr_getattr_l__mod___stages___0_____0___conv2b_weight.reshape(1, 64, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___0_____0___conv2b_gain = self.getattr_getattr_L__mod___stages___0_____0___conv2b_gain
    mul_8 = getattr_getattr_l__mod___stages___0_____0___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___0_____0___conv2b_gain = None
    view_7 = mul_8.view(-1);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_7 = torch.nn.functional.batch_norm(reshape_7, None, None, weight = view_7, training = True, momentum = 0.0, eps = 1e-05);  reshape_7 = view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_7 = batch_norm_7.reshape_as(getattr_getattr_l__mod___stages___0_____0___conv2b_weight);  batch_norm_7 = getattr_getattr_l__mod___stages___0_____0___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___0_____0___conv2b_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2b_bias
    out_3 = torch.conv2d(getattr_getattr_l__mod___stages___0_____0___act2b, weight_7, getattr_getattr_l__mod___stages___0_____0___conv2b_bias, (1, 1), (1, 1), (1, 1), 1);  getattr_getattr_l__mod___stages___0_____0___act2b = weight_7 = getattr_getattr_l__mod___stages___0_____0___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___0_____0___act3 = self.getattr_getattr_L__mod___stages___0_____0___act3(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___0_____0___conv3_weight = self.getattr_getattr_L__mod___stages___0_____0___conv3_weight
    reshape_8 = getattr_getattr_l__mod___stages___0_____0___conv3_weight.reshape(1, 256, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___0_____0___conv3_gain = self.getattr_getattr_L__mod___stages___0_____0___conv3_gain
    mul_9 = getattr_getattr_l__mod___stages___0_____0___conv3_gain * 0.22351616621017456;  getattr_getattr_l__mod___stages___0_____0___conv3_gain = None
    view_8 = mul_9.view(-1);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_8 = torch.nn.functional.batch_norm(reshape_8, None, None, weight = view_8, training = True, momentum = 0.0, eps = 1e-05);  reshape_8 = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_8 = batch_norm_8.reshape_as(getattr_getattr_l__mod___stages___0_____0___conv3_weight);  batch_norm_8 = getattr_getattr_l__mod___stages___0_____0___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___0_____0___conv3_bias = self.getattr_getattr_L__mod___stages___0_____0___conv3_bias
    out_4 = torch.conv2d(getattr_getattr_l__mod___stages___0_____0___act3, weight_8, getattr_getattr_l__mod___stages___0_____0___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___0_____0___act3 = weight_8 = getattr_getattr_l__mod___stages___0_____0___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = out_4.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_1 = self.getattr_getattr_L__mod___stages___0_____0___attn_last_fc1(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___0_____0___attn_last_bn = self.getattr_getattr_L__mod___stages___0_____0___attn_last_bn(x_se_1);  x_se_1 = None
    x_se_2 = self.getattr_getattr_L__mod___stages___0_____0___attn_last_act(getattr_getattr_l__mod___stages___0_____0___attn_last_bn);  getattr_getattr_l__mod___stages___0_____0___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_3 = self.getattr_getattr_L__mod___stages___0_____0___attn_last_fc2(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid = x_se_3.sigmoid();  x_se_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_10 = out_4 * sigmoid;  out_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_5 = 2.0 * mul_10;  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_6 = self.getattr_getattr_L__mod___stages___0_____0___drop_path(out_5);  out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_12 = out_6 * 0.2;  out_6 = None
    shortcut_2 = mul_12 + shortcut_1;  mul_12 = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___1_____0___act1 = self.getattr_getattr_L__mod___stages___1_____0___act1(shortcut_2);  shortcut_2 = None
    out_8 = getattr_getattr_l__mod___stages___1_____0___act1 * 0.9805806756909201;  getattr_getattr_l__mod___stages___1_____0___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    getattr_getattr_l__mod___stages___1_____0___downsample_pool = self.getattr_getattr_L__mod___stages___1_____0___downsample_pool(out_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____0___downsample_conv_weight = self.getattr_getattr_L__mod___stages___1_____0___downsample_conv_weight
    reshape_9 = getattr_getattr_l__mod___stages___1_____0___downsample_conv_weight.reshape(1, 512, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____0___downsample_conv_gain = self.getattr_getattr_L__mod___stages___1_____0___downsample_conv_gain
    mul_14 = getattr_getattr_l__mod___stages___1_____0___downsample_conv_gain * 0.11175808310508728;  getattr_getattr_l__mod___stages___1_____0___downsample_conv_gain = None
    view_9 = mul_14.view(-1);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_9 = torch.nn.functional.batch_norm(reshape_9, None, None, weight = view_9, training = True, momentum = 0.0, eps = 1e-05);  reshape_9 = view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_9 = batch_norm_9.reshape_as(getattr_getattr_l__mod___stages___1_____0___downsample_conv_weight);  batch_norm_9 = getattr_getattr_l__mod___stages___1_____0___downsample_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____0___downsample_conv_bias = self.getattr_getattr_L__mod___stages___1_____0___downsample_conv_bias
    shortcut_3 = torch.conv2d(getattr_getattr_l__mod___stages___1_____0___downsample_pool, weight_9, getattr_getattr_l__mod___stages___1_____0___downsample_conv_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___1_____0___downsample_pool = weight_9 = getattr_getattr_l__mod___stages___1_____0___downsample_conv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____0___conv1_weight = self.getattr_getattr_L__mod___stages___1_____0___conv1_weight
    reshape_10 = getattr_getattr_l__mod___stages___1_____0___conv1_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____0___conv1_gain = self.getattr_getattr_L__mod___stages___1_____0___conv1_gain
    mul_15 = getattr_getattr_l__mod___stages___1_____0___conv1_gain * 0.11175808310508728;  getattr_getattr_l__mod___stages___1_____0___conv1_gain = None
    view_10 = mul_15.view(-1);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_10 = torch.nn.functional.batch_norm(reshape_10, None, None, weight = view_10, training = True, momentum = 0.0, eps = 1e-05);  reshape_10 = view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_10 = batch_norm_10.reshape_as(getattr_getattr_l__mod___stages___1_____0___conv1_weight);  batch_norm_10 = getattr_getattr_l__mod___stages___1_____0___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____0___conv1_bias = self.getattr_getattr_L__mod___stages___1_____0___conv1_bias
    out_9 = torch.conv2d(out_8, weight_10, getattr_getattr_l__mod___stages___1_____0___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_8 = weight_10 = getattr_getattr_l__mod___stages___1_____0___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___1_____0___act2 = self.getattr_getattr_L__mod___stages___1_____0___act2(out_9);  out_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____0___conv2_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2_weight
    reshape_11 = getattr_getattr_l__mod___stages___1_____0___conv2_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____0___conv2_gain = self.getattr_getattr_L__mod___stages___1_____0___conv2_gain
    mul_16 = getattr_getattr_l__mod___stages___1_____0___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___1_____0___conv2_gain = None
    view_11 = mul_16.view(-1);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_11 = torch.nn.functional.batch_norm(reshape_11, None, None, weight = view_11, training = True, momentum = 0.0, eps = 1e-05);  reshape_11 = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_11 = batch_norm_11.reshape_as(getattr_getattr_l__mod___stages___1_____0___conv2_weight);  batch_norm_11 = getattr_getattr_l__mod___stages___1_____0___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____0___conv2_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2_bias
    out_10 = torch.conv2d(getattr_getattr_l__mod___stages___1_____0___act2, weight_11, getattr_getattr_l__mod___stages___1_____0___conv2_bias, (2, 2), (1, 1), (1, 1), 2);  getattr_getattr_l__mod___stages___1_____0___act2 = weight_11 = getattr_getattr_l__mod___stages___1_____0___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___1_____0___act2b = self.getattr_getattr_L__mod___stages___1_____0___act2b(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____0___conv2b_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2b_weight
    reshape_12 = getattr_getattr_l__mod___stages___1_____0___conv2b_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____0___conv2b_gain = self.getattr_getattr_L__mod___stages___1_____0___conv2b_gain
    mul_17 = getattr_getattr_l__mod___stages___1_____0___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___1_____0___conv2b_gain = None
    view_12 = mul_17.view(-1);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_12 = torch.nn.functional.batch_norm(reshape_12, None, None, weight = view_12, training = True, momentum = 0.0, eps = 1e-05);  reshape_12 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_12 = batch_norm_12.reshape_as(getattr_getattr_l__mod___stages___1_____0___conv2b_weight);  batch_norm_12 = getattr_getattr_l__mod___stages___1_____0___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____0___conv2b_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2b_bias
    out_11 = torch.conv2d(getattr_getattr_l__mod___stages___1_____0___act2b, weight_12, getattr_getattr_l__mod___stages___1_____0___conv2b_bias, (1, 1), (1, 1), (1, 1), 2);  getattr_getattr_l__mod___stages___1_____0___act2b = weight_12 = getattr_getattr_l__mod___stages___1_____0___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___1_____0___act3 = self.getattr_getattr_L__mod___stages___1_____0___act3(out_11);  out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____0___conv3_weight = self.getattr_getattr_L__mod___stages___1_____0___conv3_weight
    reshape_13 = getattr_getattr_l__mod___stages___1_____0___conv3_weight.reshape(1, 512, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____0___conv3_gain = self.getattr_getattr_L__mod___stages___1_____0___conv3_gain
    mul_18 = getattr_getattr_l__mod___stages___1_____0___conv3_gain * 0.1580497968320339;  getattr_getattr_l__mod___stages___1_____0___conv3_gain = None
    view_13 = mul_18.view(-1);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_13 = torch.nn.functional.batch_norm(reshape_13, None, None, weight = view_13, training = True, momentum = 0.0, eps = 1e-05);  reshape_13 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_13 = batch_norm_13.reshape_as(getattr_getattr_l__mod___stages___1_____0___conv3_weight);  batch_norm_13 = getattr_getattr_l__mod___stages___1_____0___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____0___conv3_bias = self.getattr_getattr_L__mod___stages___1_____0___conv3_bias
    out_12 = torch.conv2d(getattr_getattr_l__mod___stages___1_____0___act3, weight_13, getattr_getattr_l__mod___stages___1_____0___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___1_____0___act3 = weight_13 = getattr_getattr_l__mod___stages___1_____0___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = out_12.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_5 = self.getattr_getattr_L__mod___stages___1_____0___attn_last_fc1(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___1_____0___attn_last_bn = self.getattr_getattr_L__mod___stages___1_____0___attn_last_bn(x_se_5);  x_se_5 = None
    x_se_6 = self.getattr_getattr_L__mod___stages___1_____0___attn_last_act(getattr_getattr_l__mod___stages___1_____0___attn_last_bn);  getattr_getattr_l__mod___stages___1_____0___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_7 = self.getattr_getattr_L__mod___stages___1_____0___attn_last_fc2(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = x_se_7.sigmoid();  x_se_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_19 = out_12 * sigmoid_1;  out_12 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_13 = 2.0 * mul_19;  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_14 = self.getattr_getattr_L__mod___stages___1_____0___drop_path(out_13);  out_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_21 = out_14 * 0.2;  out_14 = None
    shortcut_4 = mul_21 + shortcut_3;  mul_21 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___1_____1___act1 = self.getattr_getattr_L__mod___stages___1_____1___act1(shortcut_4)
    out_16 = getattr_getattr_l__mod___stages___1_____1___act1 * 0.9805806756909201;  getattr_getattr_l__mod___stages___1_____1___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____1___conv1_weight = self.getattr_getattr_L__mod___stages___1_____1___conv1_weight
    reshape_14 = getattr_getattr_l__mod___stages___1_____1___conv1_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____1___conv1_gain = self.getattr_getattr_L__mod___stages___1_____1___conv1_gain
    mul_23 = getattr_getattr_l__mod___stages___1_____1___conv1_gain * 0.07902489841601695;  getattr_getattr_l__mod___stages___1_____1___conv1_gain = None
    view_14 = mul_23.view(-1);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_14 = torch.nn.functional.batch_norm(reshape_14, None, None, weight = view_14, training = True, momentum = 0.0, eps = 1e-05);  reshape_14 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_14 = batch_norm_14.reshape_as(getattr_getattr_l__mod___stages___1_____1___conv1_weight);  batch_norm_14 = getattr_getattr_l__mod___stages___1_____1___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____1___conv1_bias = self.getattr_getattr_L__mod___stages___1_____1___conv1_bias
    out_17 = torch.conv2d(out_16, weight_14, getattr_getattr_l__mod___stages___1_____1___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_16 = weight_14 = getattr_getattr_l__mod___stages___1_____1___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___1_____1___act2 = self.getattr_getattr_L__mod___stages___1_____1___act2(out_17);  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____1___conv2_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2_weight
    reshape_15 = getattr_getattr_l__mod___stages___1_____1___conv2_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____1___conv2_gain = self.getattr_getattr_L__mod___stages___1_____1___conv2_gain
    mul_24 = getattr_getattr_l__mod___stages___1_____1___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___1_____1___conv2_gain = None
    view_15 = mul_24.view(-1);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_15 = torch.nn.functional.batch_norm(reshape_15, None, None, weight = view_15, training = True, momentum = 0.0, eps = 1e-05);  reshape_15 = view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_15 = batch_norm_15.reshape_as(getattr_getattr_l__mod___stages___1_____1___conv2_weight);  batch_norm_15 = getattr_getattr_l__mod___stages___1_____1___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____1___conv2_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2_bias
    out_18 = torch.conv2d(getattr_getattr_l__mod___stages___1_____1___act2, weight_15, getattr_getattr_l__mod___stages___1_____1___conv2_bias, (1, 1), (1, 1), (1, 1), 2);  getattr_getattr_l__mod___stages___1_____1___act2 = weight_15 = getattr_getattr_l__mod___stages___1_____1___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___1_____1___act2b = self.getattr_getattr_L__mod___stages___1_____1___act2b(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____1___conv2b_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2b_weight
    reshape_16 = getattr_getattr_l__mod___stages___1_____1___conv2b_weight.reshape(1, 128, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____1___conv2b_gain = self.getattr_getattr_L__mod___stages___1_____1___conv2b_gain
    mul_25 = getattr_getattr_l__mod___stages___1_____1___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___1_____1___conv2b_gain = None
    view_16 = mul_25.view(-1);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_16 = torch.nn.functional.batch_norm(reshape_16, None, None, weight = view_16, training = True, momentum = 0.0, eps = 1e-05);  reshape_16 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_16 = batch_norm_16.reshape_as(getattr_getattr_l__mod___stages___1_____1___conv2b_weight);  batch_norm_16 = getattr_getattr_l__mod___stages___1_____1___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____1___conv2b_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2b_bias
    out_19 = torch.conv2d(getattr_getattr_l__mod___stages___1_____1___act2b, weight_16, getattr_getattr_l__mod___stages___1_____1___conv2b_bias, (1, 1), (1, 1), (1, 1), 2);  getattr_getattr_l__mod___stages___1_____1___act2b = weight_16 = getattr_getattr_l__mod___stages___1_____1___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___1_____1___act3 = self.getattr_getattr_L__mod___stages___1_____1___act3(out_19);  out_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___1_____1___conv3_weight = self.getattr_getattr_L__mod___stages___1_____1___conv3_weight
    reshape_17 = getattr_getattr_l__mod___stages___1_____1___conv3_weight.reshape(1, 512, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___1_____1___conv3_gain = self.getattr_getattr_L__mod___stages___1_____1___conv3_gain
    mul_26 = getattr_getattr_l__mod___stages___1_____1___conv3_gain * 0.1580497968320339;  getattr_getattr_l__mod___stages___1_____1___conv3_gain = None
    view_17 = mul_26.view(-1);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_17 = torch.nn.functional.batch_norm(reshape_17, None, None, weight = view_17, training = True, momentum = 0.0, eps = 1e-05);  reshape_17 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_17 = batch_norm_17.reshape_as(getattr_getattr_l__mod___stages___1_____1___conv3_weight);  batch_norm_17 = getattr_getattr_l__mod___stages___1_____1___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___1_____1___conv3_bias = self.getattr_getattr_L__mod___stages___1_____1___conv3_bias
    out_20 = torch.conv2d(getattr_getattr_l__mod___stages___1_____1___act3, weight_17, getattr_getattr_l__mod___stages___1_____1___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___1_____1___act3 = weight_17 = getattr_getattr_l__mod___stages___1_____1___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = out_20.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_9 = self.getattr_getattr_L__mod___stages___1_____1___attn_last_fc1(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___1_____1___attn_last_bn = self.getattr_getattr_L__mod___stages___1_____1___attn_last_bn(x_se_9);  x_se_9 = None
    x_se_10 = self.getattr_getattr_L__mod___stages___1_____1___attn_last_act(getattr_getattr_l__mod___stages___1_____1___attn_last_bn);  getattr_getattr_l__mod___stages___1_____1___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_11 = self.getattr_getattr_L__mod___stages___1_____1___attn_last_fc2(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = x_se_11.sigmoid();  x_se_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_27 = out_20 * sigmoid_2;  out_20 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_21 = 2.0 * mul_27;  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_22 = self.getattr_getattr_L__mod___stages___1_____1___drop_path(out_21);  out_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_29 = out_22 * 0.2;  out_22 = None
    shortcut_5 = mul_29 + shortcut_4;  mul_29 = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____0___act1 = self.getattr_getattr_L__mod___stages___2_____0___act1(shortcut_5);  shortcut_5 = None
    out_24 = getattr_getattr_l__mod___stages___2_____0___act1 * 0.9622504486493761;  getattr_getattr_l__mod___stages___2_____0___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    getattr_getattr_l__mod___stages___2_____0___downsample_pool = self.getattr_getattr_L__mod___stages___2_____0___downsample_pool(out_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____0___downsample_conv_weight = self.getattr_getattr_L__mod___stages___2_____0___downsample_conv_weight
    reshape_18 = getattr_getattr_l__mod___stages___2_____0___downsample_conv_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____0___downsample_conv_gain = self.getattr_getattr_L__mod___stages___2_____0___downsample_conv_gain
    mul_31 = getattr_getattr_l__mod___stages___2_____0___downsample_conv_gain * 0.07902489841601695;  getattr_getattr_l__mod___stages___2_____0___downsample_conv_gain = None
    view_18 = mul_31.view(-1);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_18 = torch.nn.functional.batch_norm(reshape_18, None, None, weight = view_18, training = True, momentum = 0.0, eps = 1e-05);  reshape_18 = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_18 = batch_norm_18.reshape_as(getattr_getattr_l__mod___stages___2_____0___downsample_conv_weight);  batch_norm_18 = getattr_getattr_l__mod___stages___2_____0___downsample_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____0___downsample_conv_bias = self.getattr_getattr_L__mod___stages___2_____0___downsample_conv_bias
    shortcut_6 = torch.conv2d(getattr_getattr_l__mod___stages___2_____0___downsample_pool, weight_18, getattr_getattr_l__mod___stages___2_____0___downsample_conv_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____0___downsample_pool = weight_18 = getattr_getattr_l__mod___stages___2_____0___downsample_conv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____0___conv1_weight = self.getattr_getattr_L__mod___stages___2_____0___conv1_weight
    reshape_19 = getattr_getattr_l__mod___stages___2_____0___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____0___conv1_gain = self.getattr_getattr_L__mod___stages___2_____0___conv1_gain
    mul_32 = getattr_getattr_l__mod___stages___2_____0___conv1_gain * 0.07902489841601695;  getattr_getattr_l__mod___stages___2_____0___conv1_gain = None
    view_19 = mul_32.view(-1);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_19 = torch.nn.functional.batch_norm(reshape_19, None, None, weight = view_19, training = True, momentum = 0.0, eps = 1e-05);  reshape_19 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_19 = batch_norm_19.reshape_as(getattr_getattr_l__mod___stages___2_____0___conv1_weight);  batch_norm_19 = getattr_getattr_l__mod___stages___2_____0___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____0___conv1_bias = self.getattr_getattr_L__mod___stages___2_____0___conv1_bias
    out_25 = torch.conv2d(out_24, weight_19, getattr_getattr_l__mod___stages___2_____0___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_24 = weight_19 = getattr_getattr_l__mod___stages___2_____0___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____0___act2 = self.getattr_getattr_L__mod___stages___2_____0___act2(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____0___conv2_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2_weight
    reshape_20 = getattr_getattr_l__mod___stages___2_____0___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____0___conv2_gain = self.getattr_getattr_L__mod___stages___2_____0___conv2_gain
    mul_33 = getattr_getattr_l__mod___stages___2_____0___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____0___conv2_gain = None
    view_20 = mul_33.view(-1);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_20 = torch.nn.functional.batch_norm(reshape_20, None, None, weight = view_20, training = True, momentum = 0.0, eps = 1e-05);  reshape_20 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_20 = batch_norm_20.reshape_as(getattr_getattr_l__mod___stages___2_____0___conv2_weight);  batch_norm_20 = getattr_getattr_l__mod___stages___2_____0___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____0___conv2_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2_bias
    out_26 = torch.conv2d(getattr_getattr_l__mod___stages___2_____0___act2, weight_20, getattr_getattr_l__mod___stages___2_____0___conv2_bias, (2, 2), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____0___act2 = weight_20 = getattr_getattr_l__mod___stages___2_____0___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____0___act2b = self.getattr_getattr_L__mod___stages___2_____0___act2b(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____0___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2b_weight
    reshape_21 = getattr_getattr_l__mod___stages___2_____0___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____0___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____0___conv2b_gain
    mul_34 = getattr_getattr_l__mod___stages___2_____0___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____0___conv2b_gain = None
    view_21 = mul_34.view(-1);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_21 = torch.nn.functional.batch_norm(reshape_21, None, None, weight = view_21, training = True, momentum = 0.0, eps = 1e-05);  reshape_21 = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_21 = batch_norm_21.reshape_as(getattr_getattr_l__mod___stages___2_____0___conv2b_weight);  batch_norm_21 = getattr_getattr_l__mod___stages___2_____0___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____0___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2b_bias
    out_27 = torch.conv2d(getattr_getattr_l__mod___stages___2_____0___act2b, weight_21, getattr_getattr_l__mod___stages___2_____0___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____0___act2b = weight_21 = getattr_getattr_l__mod___stages___2_____0___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____0___act3 = self.getattr_getattr_L__mod___stages___2_____0___act3(out_27);  out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____0___conv3_weight = self.getattr_getattr_L__mod___stages___2_____0___conv3_weight
    reshape_22 = getattr_getattr_l__mod___stages___2_____0___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____0___conv3_gain = self.getattr_getattr_L__mod___stages___2_____0___conv3_gain
    mul_35 = getattr_getattr_l__mod___stages___2_____0___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____0___conv3_gain = None
    view_22 = mul_35.view(-1);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_22 = torch.nn.functional.batch_norm(reshape_22, None, None, weight = view_22, training = True, momentum = 0.0, eps = 1e-05);  reshape_22 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_22 = batch_norm_22.reshape_as(getattr_getattr_l__mod___stages___2_____0___conv3_weight);  batch_norm_22 = getattr_getattr_l__mod___stages___2_____0___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____0___conv3_bias = self.getattr_getattr_L__mod___stages___2_____0___conv3_bias
    out_28 = torch.conv2d(getattr_getattr_l__mod___stages___2_____0___act3, weight_22, getattr_getattr_l__mod___stages___2_____0___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____0___act3 = weight_22 = getattr_getattr_l__mod___stages___2_____0___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = out_28.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_13 = self.getattr_getattr_L__mod___stages___2_____0___attn_last_fc1(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____0___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____0___attn_last_bn(x_se_13);  x_se_13 = None
    x_se_14 = self.getattr_getattr_L__mod___stages___2_____0___attn_last_act(getattr_getattr_l__mod___stages___2_____0___attn_last_bn);  getattr_getattr_l__mod___stages___2_____0___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_15 = self.getattr_getattr_L__mod___stages___2_____0___attn_last_fc2(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = x_se_15.sigmoid();  x_se_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_36 = out_28 * sigmoid_3;  out_28 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_29 = 2.0 * mul_36;  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_30 = self.getattr_getattr_L__mod___stages___2_____0___drop_path(out_29);  out_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_38 = out_30 * 0.2;  out_30 = None
    shortcut_7 = mul_38 + shortcut_6;  mul_38 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____1___act1 = self.getattr_getattr_L__mod___stages___2_____1___act1(shortcut_7)
    out_32 = getattr_getattr_l__mod___stages___2_____1___act1 * 0.9805806756909201;  getattr_getattr_l__mod___stages___2_____1___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____1___conv1_weight = self.getattr_getattr_L__mod___stages___2_____1___conv1_weight
    reshape_23 = getattr_getattr_l__mod___stages___2_____1___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____1___conv1_gain = self.getattr_getattr_L__mod___stages___2_____1___conv1_gain
    mul_40 = getattr_getattr_l__mod___stages___2_____1___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___2_____1___conv1_gain = None
    view_23 = mul_40.view(-1);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_23 = torch.nn.functional.batch_norm(reshape_23, None, None, weight = view_23, training = True, momentum = 0.0, eps = 1e-05);  reshape_23 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_23 = batch_norm_23.reshape_as(getattr_getattr_l__mod___stages___2_____1___conv1_weight);  batch_norm_23 = getattr_getattr_l__mod___stages___2_____1___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____1___conv1_bias = self.getattr_getattr_L__mod___stages___2_____1___conv1_bias
    out_33 = torch.conv2d(out_32, weight_23, getattr_getattr_l__mod___stages___2_____1___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_32 = weight_23 = getattr_getattr_l__mod___stages___2_____1___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____1___act2 = self.getattr_getattr_L__mod___stages___2_____1___act2(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____1___conv2_weight = self.getattr_getattr_L__mod___stages___2_____1___conv2_weight
    reshape_24 = getattr_getattr_l__mod___stages___2_____1___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____1___conv2_gain = self.getattr_getattr_L__mod___stages___2_____1___conv2_gain
    mul_41 = getattr_getattr_l__mod___stages___2_____1___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____1___conv2_gain = None
    view_24 = mul_41.view(-1);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_24 = torch.nn.functional.batch_norm(reshape_24, None, None, weight = view_24, training = True, momentum = 0.0, eps = 1e-05);  reshape_24 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_24 = batch_norm_24.reshape_as(getattr_getattr_l__mod___stages___2_____1___conv2_weight);  batch_norm_24 = getattr_getattr_l__mod___stages___2_____1___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____1___conv2_bias = self.getattr_getattr_L__mod___stages___2_____1___conv2_bias
    out_34 = torch.conv2d(getattr_getattr_l__mod___stages___2_____1___act2, weight_24, getattr_getattr_l__mod___stages___2_____1___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____1___act2 = weight_24 = getattr_getattr_l__mod___stages___2_____1___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____1___act2b = self.getattr_getattr_L__mod___stages___2_____1___act2b(out_34);  out_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____1___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____1___conv2b_weight
    reshape_25 = getattr_getattr_l__mod___stages___2_____1___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____1___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____1___conv2b_gain
    mul_42 = getattr_getattr_l__mod___stages___2_____1___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____1___conv2b_gain = None
    view_25 = mul_42.view(-1);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_25 = torch.nn.functional.batch_norm(reshape_25, None, None, weight = view_25, training = True, momentum = 0.0, eps = 1e-05);  reshape_25 = view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_25 = batch_norm_25.reshape_as(getattr_getattr_l__mod___stages___2_____1___conv2b_weight);  batch_norm_25 = getattr_getattr_l__mod___stages___2_____1___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____1___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____1___conv2b_bias
    out_35 = torch.conv2d(getattr_getattr_l__mod___stages___2_____1___act2b, weight_25, getattr_getattr_l__mod___stages___2_____1___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____1___act2b = weight_25 = getattr_getattr_l__mod___stages___2_____1___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____1___act3 = self.getattr_getattr_L__mod___stages___2_____1___act3(out_35);  out_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____1___conv3_weight = self.getattr_getattr_L__mod___stages___2_____1___conv3_weight
    reshape_26 = getattr_getattr_l__mod___stages___2_____1___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____1___conv3_gain = self.getattr_getattr_L__mod___stages___2_____1___conv3_gain
    mul_43 = getattr_getattr_l__mod___stages___2_____1___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____1___conv3_gain = None
    view_26 = mul_43.view(-1);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_26 = torch.nn.functional.batch_norm(reshape_26, None, None, weight = view_26, training = True, momentum = 0.0, eps = 1e-05);  reshape_26 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_26 = batch_norm_26.reshape_as(getattr_getattr_l__mod___stages___2_____1___conv3_weight);  batch_norm_26 = getattr_getattr_l__mod___stages___2_____1___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____1___conv3_bias = self.getattr_getattr_L__mod___stages___2_____1___conv3_bias
    out_36 = torch.conv2d(getattr_getattr_l__mod___stages___2_____1___act3, weight_26, getattr_getattr_l__mod___stages___2_____1___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____1___act3 = weight_26 = getattr_getattr_l__mod___stages___2_____1___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = out_36.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_17 = self.getattr_getattr_L__mod___stages___2_____1___attn_last_fc1(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____1___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____1___attn_last_bn(x_se_17);  x_se_17 = None
    x_se_18 = self.getattr_getattr_L__mod___stages___2_____1___attn_last_act(getattr_getattr_l__mod___stages___2_____1___attn_last_bn);  getattr_getattr_l__mod___stages___2_____1___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_19 = self.getattr_getattr_L__mod___stages___2_____1___attn_last_fc2(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = x_se_19.sigmoid();  x_se_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_44 = out_36 * sigmoid_4;  out_36 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_37 = 2.0 * mul_44;  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_38 = self.getattr_getattr_L__mod___stages___2_____1___drop_path(out_37);  out_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_46 = out_38 * 0.2;  out_38 = None
    shortcut_8 = mul_46 + shortcut_7;  mul_46 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____2___act1 = self.getattr_getattr_L__mod___stages___2_____2___act1(shortcut_8)
    out_40 = getattr_getattr_l__mod___stages___2_____2___act1 * 0.9622504486493761;  getattr_getattr_l__mod___stages___2_____2___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____2___conv1_weight = self.getattr_getattr_L__mod___stages___2_____2___conv1_weight
    reshape_27 = getattr_getattr_l__mod___stages___2_____2___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____2___conv1_gain = self.getattr_getattr_L__mod___stages___2_____2___conv1_gain
    mul_48 = getattr_getattr_l__mod___stages___2_____2___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___2_____2___conv1_gain = None
    view_27 = mul_48.view(-1);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_27 = torch.nn.functional.batch_norm(reshape_27, None, None, weight = view_27, training = True, momentum = 0.0, eps = 1e-05);  reshape_27 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_27 = batch_norm_27.reshape_as(getattr_getattr_l__mod___stages___2_____2___conv1_weight);  batch_norm_27 = getattr_getattr_l__mod___stages___2_____2___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____2___conv1_bias = self.getattr_getattr_L__mod___stages___2_____2___conv1_bias
    out_41 = torch.conv2d(out_40, weight_27, getattr_getattr_l__mod___stages___2_____2___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_40 = weight_27 = getattr_getattr_l__mod___stages___2_____2___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____2___act2 = self.getattr_getattr_L__mod___stages___2_____2___act2(out_41);  out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____2___conv2_weight = self.getattr_getattr_L__mod___stages___2_____2___conv2_weight
    reshape_28 = getattr_getattr_l__mod___stages___2_____2___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____2___conv2_gain = self.getattr_getattr_L__mod___stages___2_____2___conv2_gain
    mul_49 = getattr_getattr_l__mod___stages___2_____2___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____2___conv2_gain = None
    view_28 = mul_49.view(-1);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_28 = torch.nn.functional.batch_norm(reshape_28, None, None, weight = view_28, training = True, momentum = 0.0, eps = 1e-05);  reshape_28 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_28 = batch_norm_28.reshape_as(getattr_getattr_l__mod___stages___2_____2___conv2_weight);  batch_norm_28 = getattr_getattr_l__mod___stages___2_____2___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____2___conv2_bias = self.getattr_getattr_L__mod___stages___2_____2___conv2_bias
    out_42 = torch.conv2d(getattr_getattr_l__mod___stages___2_____2___act2, weight_28, getattr_getattr_l__mod___stages___2_____2___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____2___act2 = weight_28 = getattr_getattr_l__mod___stages___2_____2___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____2___act2b = self.getattr_getattr_L__mod___stages___2_____2___act2b(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____2___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____2___conv2b_weight
    reshape_29 = getattr_getattr_l__mod___stages___2_____2___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____2___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____2___conv2b_gain
    mul_50 = getattr_getattr_l__mod___stages___2_____2___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____2___conv2b_gain = None
    view_29 = mul_50.view(-1);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_29 = torch.nn.functional.batch_norm(reshape_29, None, None, weight = view_29, training = True, momentum = 0.0, eps = 1e-05);  reshape_29 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_29 = batch_norm_29.reshape_as(getattr_getattr_l__mod___stages___2_____2___conv2b_weight);  batch_norm_29 = getattr_getattr_l__mod___stages___2_____2___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____2___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____2___conv2b_bias
    out_43 = torch.conv2d(getattr_getattr_l__mod___stages___2_____2___act2b, weight_29, getattr_getattr_l__mod___stages___2_____2___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____2___act2b = weight_29 = getattr_getattr_l__mod___stages___2_____2___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____2___act3 = self.getattr_getattr_L__mod___stages___2_____2___act3(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____2___conv3_weight = self.getattr_getattr_L__mod___stages___2_____2___conv3_weight
    reshape_30 = getattr_getattr_l__mod___stages___2_____2___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____2___conv3_gain = self.getattr_getattr_L__mod___stages___2_____2___conv3_gain
    mul_51 = getattr_getattr_l__mod___stages___2_____2___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____2___conv3_gain = None
    view_30 = mul_51.view(-1);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_30 = torch.nn.functional.batch_norm(reshape_30, None, None, weight = view_30, training = True, momentum = 0.0, eps = 1e-05);  reshape_30 = view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_30 = batch_norm_30.reshape_as(getattr_getattr_l__mod___stages___2_____2___conv3_weight);  batch_norm_30 = getattr_getattr_l__mod___stages___2_____2___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____2___conv3_bias = self.getattr_getattr_L__mod___stages___2_____2___conv3_bias
    out_44 = torch.conv2d(getattr_getattr_l__mod___stages___2_____2___act3, weight_30, getattr_getattr_l__mod___stages___2_____2___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____2___act3 = weight_30 = getattr_getattr_l__mod___stages___2_____2___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = out_44.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_21 = self.getattr_getattr_L__mod___stages___2_____2___attn_last_fc1(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____2___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____2___attn_last_bn(x_se_21);  x_se_21 = None
    x_se_22 = self.getattr_getattr_L__mod___stages___2_____2___attn_last_act(getattr_getattr_l__mod___stages___2_____2___attn_last_bn);  getattr_getattr_l__mod___stages___2_____2___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_23 = self.getattr_getattr_L__mod___stages___2_____2___attn_last_fc2(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5 = x_se_23.sigmoid();  x_se_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_52 = out_44 * sigmoid_5;  out_44 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_45 = 2.0 * mul_52;  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_46 = self.getattr_getattr_L__mod___stages___2_____2___drop_path(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_54 = out_46 * 0.2;  out_46 = None
    shortcut_9 = mul_54 + shortcut_8;  mul_54 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____3___act1 = self.getattr_getattr_L__mod___stages___2_____3___act1(shortcut_9)
    out_48 = getattr_getattr_l__mod___stages___2_____3___act1 * 0.9449111825230679;  getattr_getattr_l__mod___stages___2_____3___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____3___conv1_weight = self.getattr_getattr_L__mod___stages___2_____3___conv1_weight
    reshape_31 = getattr_getattr_l__mod___stages___2_____3___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____3___conv1_gain = self.getattr_getattr_L__mod___stages___2_____3___conv1_gain
    mul_56 = getattr_getattr_l__mod___stages___2_____3___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___2_____3___conv1_gain = None
    view_31 = mul_56.view(-1);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_31 = torch.nn.functional.batch_norm(reshape_31, None, None, weight = view_31, training = True, momentum = 0.0, eps = 1e-05);  reshape_31 = view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_31 = batch_norm_31.reshape_as(getattr_getattr_l__mod___stages___2_____3___conv1_weight);  batch_norm_31 = getattr_getattr_l__mod___stages___2_____3___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____3___conv1_bias = self.getattr_getattr_L__mod___stages___2_____3___conv1_bias
    out_49 = torch.conv2d(out_48, weight_31, getattr_getattr_l__mod___stages___2_____3___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_48 = weight_31 = getattr_getattr_l__mod___stages___2_____3___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____3___act2 = self.getattr_getattr_L__mod___stages___2_____3___act2(out_49);  out_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____3___conv2_weight = self.getattr_getattr_L__mod___stages___2_____3___conv2_weight
    reshape_32 = getattr_getattr_l__mod___stages___2_____3___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____3___conv2_gain = self.getattr_getattr_L__mod___stages___2_____3___conv2_gain
    mul_57 = getattr_getattr_l__mod___stages___2_____3___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____3___conv2_gain = None
    view_32 = mul_57.view(-1);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_32 = torch.nn.functional.batch_norm(reshape_32, None, None, weight = view_32, training = True, momentum = 0.0, eps = 1e-05);  reshape_32 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_32 = batch_norm_32.reshape_as(getattr_getattr_l__mod___stages___2_____3___conv2_weight);  batch_norm_32 = getattr_getattr_l__mod___stages___2_____3___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____3___conv2_bias = self.getattr_getattr_L__mod___stages___2_____3___conv2_bias
    out_50 = torch.conv2d(getattr_getattr_l__mod___stages___2_____3___act2, weight_32, getattr_getattr_l__mod___stages___2_____3___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____3___act2 = weight_32 = getattr_getattr_l__mod___stages___2_____3___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____3___act2b = self.getattr_getattr_L__mod___stages___2_____3___act2b(out_50);  out_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____3___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____3___conv2b_weight
    reshape_33 = getattr_getattr_l__mod___stages___2_____3___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____3___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____3___conv2b_gain
    mul_58 = getattr_getattr_l__mod___stages___2_____3___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____3___conv2b_gain = None
    view_33 = mul_58.view(-1);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_33 = torch.nn.functional.batch_norm(reshape_33, None, None, weight = view_33, training = True, momentum = 0.0, eps = 1e-05);  reshape_33 = view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_33 = batch_norm_33.reshape_as(getattr_getattr_l__mod___stages___2_____3___conv2b_weight);  batch_norm_33 = getattr_getattr_l__mod___stages___2_____3___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____3___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____3___conv2b_bias
    out_51 = torch.conv2d(getattr_getattr_l__mod___stages___2_____3___act2b, weight_33, getattr_getattr_l__mod___stages___2_____3___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____3___act2b = weight_33 = getattr_getattr_l__mod___stages___2_____3___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____3___act3 = self.getattr_getattr_L__mod___stages___2_____3___act3(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____3___conv3_weight = self.getattr_getattr_L__mod___stages___2_____3___conv3_weight
    reshape_34 = getattr_getattr_l__mod___stages___2_____3___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____3___conv3_gain = self.getattr_getattr_L__mod___stages___2_____3___conv3_gain
    mul_59 = getattr_getattr_l__mod___stages___2_____3___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____3___conv3_gain = None
    view_34 = mul_59.view(-1);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_34 = torch.nn.functional.batch_norm(reshape_34, None, None, weight = view_34, training = True, momentum = 0.0, eps = 1e-05);  reshape_34 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_34 = batch_norm_34.reshape_as(getattr_getattr_l__mod___stages___2_____3___conv3_weight);  batch_norm_34 = getattr_getattr_l__mod___stages___2_____3___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____3___conv3_bias = self.getattr_getattr_L__mod___stages___2_____3___conv3_bias
    out_52 = torch.conv2d(getattr_getattr_l__mod___stages___2_____3___act3, weight_34, getattr_getattr_l__mod___stages___2_____3___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____3___act3 = weight_34 = getattr_getattr_l__mod___stages___2_____3___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = out_52.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_25 = self.getattr_getattr_L__mod___stages___2_____3___attn_last_fc1(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____3___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____3___attn_last_bn(x_se_25);  x_se_25 = None
    x_se_26 = self.getattr_getattr_L__mod___stages___2_____3___attn_last_act(getattr_getattr_l__mod___stages___2_____3___attn_last_bn);  getattr_getattr_l__mod___stages___2_____3___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_27 = self.getattr_getattr_L__mod___stages___2_____3___attn_last_fc2(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6 = x_se_27.sigmoid();  x_se_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_60 = out_52 * sigmoid_6;  out_52 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_53 = 2.0 * mul_60;  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_54 = self.getattr_getattr_L__mod___stages___2_____3___drop_path(out_53);  out_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_62 = out_54 * 0.2;  out_54 = None
    shortcut_10 = mul_62 + shortcut_9;  mul_62 = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____4___act1 = self.getattr_getattr_L__mod___stages___2_____4___act1(shortcut_10)
    out_56 = getattr_getattr_l__mod___stages___2_____4___act1 * 0.9284766908852592;  getattr_getattr_l__mod___stages___2_____4___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____4___conv1_weight = self.getattr_getattr_L__mod___stages___2_____4___conv1_weight
    reshape_35 = getattr_getattr_l__mod___stages___2_____4___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____4___conv1_gain = self.getattr_getattr_L__mod___stages___2_____4___conv1_gain
    mul_64 = getattr_getattr_l__mod___stages___2_____4___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___2_____4___conv1_gain = None
    view_35 = mul_64.view(-1);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_35 = torch.nn.functional.batch_norm(reshape_35, None, None, weight = view_35, training = True, momentum = 0.0, eps = 1e-05);  reshape_35 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_35 = batch_norm_35.reshape_as(getattr_getattr_l__mod___stages___2_____4___conv1_weight);  batch_norm_35 = getattr_getattr_l__mod___stages___2_____4___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____4___conv1_bias = self.getattr_getattr_L__mod___stages___2_____4___conv1_bias
    out_57 = torch.conv2d(out_56, weight_35, getattr_getattr_l__mod___stages___2_____4___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_56 = weight_35 = getattr_getattr_l__mod___stages___2_____4___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____4___act2 = self.getattr_getattr_L__mod___stages___2_____4___act2(out_57);  out_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____4___conv2_weight = self.getattr_getattr_L__mod___stages___2_____4___conv2_weight
    reshape_36 = getattr_getattr_l__mod___stages___2_____4___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____4___conv2_gain = self.getattr_getattr_L__mod___stages___2_____4___conv2_gain
    mul_65 = getattr_getattr_l__mod___stages___2_____4___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____4___conv2_gain = None
    view_36 = mul_65.view(-1);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_36 = torch.nn.functional.batch_norm(reshape_36, None, None, weight = view_36, training = True, momentum = 0.0, eps = 1e-05);  reshape_36 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_36 = batch_norm_36.reshape_as(getattr_getattr_l__mod___stages___2_____4___conv2_weight);  batch_norm_36 = getattr_getattr_l__mod___stages___2_____4___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____4___conv2_bias = self.getattr_getattr_L__mod___stages___2_____4___conv2_bias
    out_58 = torch.conv2d(getattr_getattr_l__mod___stages___2_____4___act2, weight_36, getattr_getattr_l__mod___stages___2_____4___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____4___act2 = weight_36 = getattr_getattr_l__mod___stages___2_____4___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____4___act2b = self.getattr_getattr_L__mod___stages___2_____4___act2b(out_58);  out_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____4___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____4___conv2b_weight
    reshape_37 = getattr_getattr_l__mod___stages___2_____4___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____4___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____4___conv2b_gain
    mul_66 = getattr_getattr_l__mod___stages___2_____4___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____4___conv2b_gain = None
    view_37 = mul_66.view(-1);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_37 = torch.nn.functional.batch_norm(reshape_37, None, None, weight = view_37, training = True, momentum = 0.0, eps = 1e-05);  reshape_37 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_37 = batch_norm_37.reshape_as(getattr_getattr_l__mod___stages___2_____4___conv2b_weight);  batch_norm_37 = getattr_getattr_l__mod___stages___2_____4___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____4___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____4___conv2b_bias
    out_59 = torch.conv2d(getattr_getattr_l__mod___stages___2_____4___act2b, weight_37, getattr_getattr_l__mod___stages___2_____4___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____4___act2b = weight_37 = getattr_getattr_l__mod___stages___2_____4___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____4___act3 = self.getattr_getattr_L__mod___stages___2_____4___act3(out_59);  out_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____4___conv3_weight = self.getattr_getattr_L__mod___stages___2_____4___conv3_weight
    reshape_38 = getattr_getattr_l__mod___stages___2_____4___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____4___conv3_gain = self.getattr_getattr_L__mod___stages___2_____4___conv3_gain
    mul_67 = getattr_getattr_l__mod___stages___2_____4___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____4___conv3_gain = None
    view_38 = mul_67.view(-1);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_38 = torch.nn.functional.batch_norm(reshape_38, None, None, weight = view_38, training = True, momentum = 0.0, eps = 1e-05);  reshape_38 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_38 = batch_norm_38.reshape_as(getattr_getattr_l__mod___stages___2_____4___conv3_weight);  batch_norm_38 = getattr_getattr_l__mod___stages___2_____4___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____4___conv3_bias = self.getattr_getattr_L__mod___stages___2_____4___conv3_bias
    out_60 = torch.conv2d(getattr_getattr_l__mod___stages___2_____4___act3, weight_38, getattr_getattr_l__mod___stages___2_____4___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____4___act3 = weight_38 = getattr_getattr_l__mod___stages___2_____4___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = out_60.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_29 = self.getattr_getattr_L__mod___stages___2_____4___attn_last_fc1(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____4___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____4___attn_last_bn(x_se_29);  x_se_29 = None
    x_se_30 = self.getattr_getattr_L__mod___stages___2_____4___attn_last_act(getattr_getattr_l__mod___stages___2_____4___attn_last_bn);  getattr_getattr_l__mod___stages___2_____4___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_31 = self.getattr_getattr_L__mod___stages___2_____4___attn_last_fc2(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7 = x_se_31.sigmoid();  x_se_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_68 = out_60 * sigmoid_7;  out_60 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_61 = 2.0 * mul_68;  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_62 = self.getattr_getattr_L__mod___stages___2_____4___drop_path(out_61);  out_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_70 = out_62 * 0.2;  out_62 = None
    shortcut_11 = mul_70 + shortcut_10;  mul_70 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___2_____5___act1 = self.getattr_getattr_L__mod___stages___2_____5___act1(shortcut_11)
    out_64 = getattr_getattr_l__mod___stages___2_____5___act1 * 0.9128709291752768;  getattr_getattr_l__mod___stages___2_____5___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____5___conv1_weight = self.getattr_getattr_L__mod___stages___2_____5___conv1_weight
    reshape_39 = getattr_getattr_l__mod___stages___2_____5___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____5___conv1_gain = self.getattr_getattr_L__mod___stages___2_____5___conv1_gain
    mul_72 = getattr_getattr_l__mod___stages___2_____5___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___2_____5___conv1_gain = None
    view_39 = mul_72.view(-1);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_39 = torch.nn.functional.batch_norm(reshape_39, None, None, weight = view_39, training = True, momentum = 0.0, eps = 1e-05);  reshape_39 = view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_39 = batch_norm_39.reshape_as(getattr_getattr_l__mod___stages___2_____5___conv1_weight);  batch_norm_39 = getattr_getattr_l__mod___stages___2_____5___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____5___conv1_bias = self.getattr_getattr_L__mod___stages___2_____5___conv1_bias
    out_65 = torch.conv2d(out_64, weight_39, getattr_getattr_l__mod___stages___2_____5___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_64 = weight_39 = getattr_getattr_l__mod___stages___2_____5___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___2_____5___act2 = self.getattr_getattr_L__mod___stages___2_____5___act2(out_65);  out_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____5___conv2_weight = self.getattr_getattr_L__mod___stages___2_____5___conv2_weight
    reshape_40 = getattr_getattr_l__mod___stages___2_____5___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____5___conv2_gain = self.getattr_getattr_L__mod___stages___2_____5___conv2_gain
    mul_73 = getattr_getattr_l__mod___stages___2_____5___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____5___conv2_gain = None
    view_40 = mul_73.view(-1);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_40 = torch.nn.functional.batch_norm(reshape_40, None, None, weight = view_40, training = True, momentum = 0.0, eps = 1e-05);  reshape_40 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_40 = batch_norm_40.reshape_as(getattr_getattr_l__mod___stages___2_____5___conv2_weight);  batch_norm_40 = getattr_getattr_l__mod___stages___2_____5___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____5___conv2_bias = self.getattr_getattr_L__mod___stages___2_____5___conv2_bias
    out_66 = torch.conv2d(getattr_getattr_l__mod___stages___2_____5___act2, weight_40, getattr_getattr_l__mod___stages___2_____5___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____5___act2 = weight_40 = getattr_getattr_l__mod___stages___2_____5___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___2_____5___act2b = self.getattr_getattr_L__mod___stages___2_____5___act2b(out_66);  out_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____5___conv2b_weight = self.getattr_getattr_L__mod___stages___2_____5___conv2b_weight
    reshape_41 = getattr_getattr_l__mod___stages___2_____5___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____5___conv2b_gain = self.getattr_getattr_L__mod___stages___2_____5___conv2b_gain
    mul_74 = getattr_getattr_l__mod___stages___2_____5___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___2_____5___conv2b_gain = None
    view_41 = mul_74.view(-1);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_41 = torch.nn.functional.batch_norm(reshape_41, None, None, weight = view_41, training = True, momentum = 0.0, eps = 1e-05);  reshape_41 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_41 = batch_norm_41.reshape_as(getattr_getattr_l__mod___stages___2_____5___conv2b_weight);  batch_norm_41 = getattr_getattr_l__mod___stages___2_____5___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____5___conv2b_bias = self.getattr_getattr_L__mod___stages___2_____5___conv2b_bias
    out_67 = torch.conv2d(getattr_getattr_l__mod___stages___2_____5___act2b, weight_41, getattr_getattr_l__mod___stages___2_____5___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___2_____5___act2b = weight_41 = getattr_getattr_l__mod___stages___2_____5___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___2_____5___act3 = self.getattr_getattr_L__mod___stages___2_____5___act3(out_67);  out_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___2_____5___conv3_weight = self.getattr_getattr_L__mod___stages___2_____5___conv3_weight
    reshape_42 = getattr_getattr_l__mod___stages___2_____5___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___2_____5___conv3_gain = self.getattr_getattr_L__mod___stages___2_____5___conv3_gain
    mul_75 = getattr_getattr_l__mod___stages___2_____5___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___2_____5___conv3_gain = None
    view_42 = mul_75.view(-1);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_42 = torch.nn.functional.batch_norm(reshape_42, None, None, weight = view_42, training = True, momentum = 0.0, eps = 1e-05);  reshape_42 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_42 = batch_norm_42.reshape_as(getattr_getattr_l__mod___stages___2_____5___conv3_weight);  batch_norm_42 = getattr_getattr_l__mod___stages___2_____5___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___2_____5___conv3_bias = self.getattr_getattr_L__mod___stages___2_____5___conv3_bias
    out_68 = torch.conv2d(getattr_getattr_l__mod___stages___2_____5___act3, weight_42, getattr_getattr_l__mod___stages___2_____5___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___2_____5___act3 = weight_42 = getattr_getattr_l__mod___stages___2_____5___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = out_68.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_33 = self.getattr_getattr_L__mod___stages___2_____5___attn_last_fc1(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____5___attn_last_bn = self.getattr_getattr_L__mod___stages___2_____5___attn_last_bn(x_se_33);  x_se_33 = None
    x_se_34 = self.getattr_getattr_L__mod___stages___2_____5___attn_last_act(getattr_getattr_l__mod___stages___2_____5___attn_last_bn);  getattr_getattr_l__mod___stages___2_____5___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_35 = self.getattr_getattr_L__mod___stages___2_____5___attn_last_fc2(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8 = x_se_35.sigmoid();  x_se_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_76 = out_68 * sigmoid_8;  out_68 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_69 = 2.0 * mul_76;  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_70 = self.getattr_getattr_L__mod___stages___2_____5___drop_path(out_69);  out_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_78 = out_70 * 0.2;  out_70 = None
    shortcut_12 = mul_78 + shortcut_11;  mul_78 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___3_____0___act1 = self.getattr_getattr_L__mod___stages___3_____0___act1(shortcut_12);  shortcut_12 = None
    out_72 = getattr_getattr_l__mod___stages___3_____0___act1 * 0.8980265101338745;  getattr_getattr_l__mod___stages___3_____0___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    getattr_getattr_l__mod___stages___3_____0___downsample_pool = self.getattr_getattr_L__mod___stages___3_____0___downsample_pool(out_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____0___downsample_conv_weight = self.getattr_getattr_L__mod___stages___3_____0___downsample_conv_weight
    reshape_43 = getattr_getattr_l__mod___stages___3_____0___downsample_conv_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____0___downsample_conv_gain = self.getattr_getattr_L__mod___stages___3_____0___downsample_conv_gain
    mul_80 = getattr_getattr_l__mod___stages___3_____0___downsample_conv_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___3_____0___downsample_conv_gain = None
    view_43 = mul_80.view(-1);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_43 = torch.nn.functional.batch_norm(reshape_43, None, None, weight = view_43, training = True, momentum = 0.0, eps = 1e-05);  reshape_43 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_43 = batch_norm_43.reshape_as(getattr_getattr_l__mod___stages___3_____0___downsample_conv_weight);  batch_norm_43 = getattr_getattr_l__mod___stages___3_____0___downsample_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____0___downsample_conv_bias = self.getattr_getattr_L__mod___stages___3_____0___downsample_conv_bias
    shortcut_13 = torch.conv2d(getattr_getattr_l__mod___stages___3_____0___downsample_pool, weight_43, getattr_getattr_l__mod___stages___3_____0___downsample_conv_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___3_____0___downsample_pool = weight_43 = getattr_getattr_l__mod___stages___3_____0___downsample_conv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____0___conv1_weight = self.getattr_getattr_L__mod___stages___3_____0___conv1_weight
    reshape_44 = getattr_getattr_l__mod___stages___3_____0___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____0___conv1_gain = self.getattr_getattr_L__mod___stages___3_____0___conv1_gain
    mul_81 = getattr_getattr_l__mod___stages___3_____0___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___3_____0___conv1_gain = None
    view_44 = mul_81.view(-1);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_44 = torch.nn.functional.batch_norm(reshape_44, None, None, weight = view_44, training = True, momentum = 0.0, eps = 1e-05);  reshape_44 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_44 = batch_norm_44.reshape_as(getattr_getattr_l__mod___stages___3_____0___conv1_weight);  batch_norm_44 = getattr_getattr_l__mod___stages___3_____0___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____0___conv1_bias = self.getattr_getattr_L__mod___stages___3_____0___conv1_bias
    out_73 = torch.conv2d(out_72, weight_44, getattr_getattr_l__mod___stages___3_____0___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_72 = weight_44 = getattr_getattr_l__mod___stages___3_____0___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___3_____0___act2 = self.getattr_getattr_L__mod___stages___3_____0___act2(out_73);  out_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____0___conv2_weight = self.getattr_getattr_L__mod___stages___3_____0___conv2_weight
    reshape_45 = getattr_getattr_l__mod___stages___3_____0___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____0___conv2_gain = self.getattr_getattr_L__mod___stages___3_____0___conv2_gain
    mul_82 = getattr_getattr_l__mod___stages___3_____0___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____0___conv2_gain = None
    view_45 = mul_82.view(-1);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_45 = torch.nn.functional.batch_norm(reshape_45, None, None, weight = view_45, training = True, momentum = 0.0, eps = 1e-05);  reshape_45 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_45 = batch_norm_45.reshape_as(getattr_getattr_l__mod___stages___3_____0___conv2_weight);  batch_norm_45 = getattr_getattr_l__mod___stages___3_____0___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____0___conv2_bias = self.getattr_getattr_L__mod___stages___3_____0___conv2_bias
    out_74 = torch.conv2d(getattr_getattr_l__mod___stages___3_____0___act2, weight_45, getattr_getattr_l__mod___stages___3_____0___conv2_bias, (2, 2), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____0___act2 = weight_45 = getattr_getattr_l__mod___stages___3_____0___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___3_____0___act2b = self.getattr_getattr_L__mod___stages___3_____0___act2b(out_74);  out_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____0___conv2b_weight = self.getattr_getattr_L__mod___stages___3_____0___conv2b_weight
    reshape_46 = getattr_getattr_l__mod___stages___3_____0___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____0___conv2b_gain = self.getattr_getattr_L__mod___stages___3_____0___conv2b_gain
    mul_83 = getattr_getattr_l__mod___stages___3_____0___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____0___conv2b_gain = None
    view_46 = mul_83.view(-1);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_46 = torch.nn.functional.batch_norm(reshape_46, None, None, weight = view_46, training = True, momentum = 0.0, eps = 1e-05);  reshape_46 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_46 = batch_norm_46.reshape_as(getattr_getattr_l__mod___stages___3_____0___conv2b_weight);  batch_norm_46 = getattr_getattr_l__mod___stages___3_____0___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____0___conv2b_bias = self.getattr_getattr_L__mod___stages___3_____0___conv2b_bias
    out_75 = torch.conv2d(getattr_getattr_l__mod___stages___3_____0___act2b, weight_46, getattr_getattr_l__mod___stages___3_____0___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____0___act2b = weight_46 = getattr_getattr_l__mod___stages___3_____0___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___3_____0___act3 = self.getattr_getattr_L__mod___stages___3_____0___act3(out_75);  out_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____0___conv3_weight = self.getattr_getattr_L__mod___stages___3_____0___conv3_weight
    reshape_47 = getattr_getattr_l__mod___stages___3_____0___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____0___conv3_gain = self.getattr_getattr_L__mod___stages___3_____0___conv3_gain
    mul_84 = getattr_getattr_l__mod___stages___3_____0___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___3_____0___conv3_gain = None
    view_47 = mul_84.view(-1);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_47 = torch.nn.functional.batch_norm(reshape_47, None, None, weight = view_47, training = True, momentum = 0.0, eps = 1e-05);  reshape_47 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_47 = batch_norm_47.reshape_as(getattr_getattr_l__mod___stages___3_____0___conv3_weight);  batch_norm_47 = getattr_getattr_l__mod___stages___3_____0___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____0___conv3_bias = self.getattr_getattr_L__mod___stages___3_____0___conv3_bias
    out_76 = torch.conv2d(getattr_getattr_l__mod___stages___3_____0___act3, weight_47, getattr_getattr_l__mod___stages___3_____0___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___3_____0___act3 = weight_47 = getattr_getattr_l__mod___stages___3_____0___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = out_76.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_37 = self.getattr_getattr_L__mod___stages___3_____0___attn_last_fc1(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___3_____0___attn_last_bn = self.getattr_getattr_L__mod___stages___3_____0___attn_last_bn(x_se_37);  x_se_37 = None
    x_se_38 = self.getattr_getattr_L__mod___stages___3_____0___attn_last_act(getattr_getattr_l__mod___stages___3_____0___attn_last_bn);  getattr_getattr_l__mod___stages___3_____0___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_39 = self.getattr_getattr_L__mod___stages___3_____0___attn_last_fc2(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9 = x_se_39.sigmoid();  x_se_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_85 = out_76 * sigmoid_9;  out_76 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_77 = 2.0 * mul_85;  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_78 = self.getattr_getattr_L__mod___stages___3_____0___drop_path(out_77);  out_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_87 = out_78 * 0.2;  out_78 = None
    shortcut_14 = mul_87 + shortcut_13;  mul_87 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___3_____1___act1 = self.getattr_getattr_L__mod___stages___3_____1___act1(shortcut_14)
    out_80 = getattr_getattr_l__mod___stages___3_____1___act1 * 0.9805806756909201;  getattr_getattr_l__mod___stages___3_____1___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____1___conv1_weight = self.getattr_getattr_L__mod___stages___3_____1___conv1_weight
    reshape_48 = getattr_getattr_l__mod___stages___3_____1___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____1___conv1_gain = self.getattr_getattr_L__mod___stages___3_____1___conv1_gain
    mul_89 = getattr_getattr_l__mod___stages___3_____1___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___3_____1___conv1_gain = None
    view_48 = mul_89.view(-1);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_48 = torch.nn.functional.batch_norm(reshape_48, None, None, weight = view_48, training = True, momentum = 0.0, eps = 1e-05);  reshape_48 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_48 = batch_norm_48.reshape_as(getattr_getattr_l__mod___stages___3_____1___conv1_weight);  batch_norm_48 = getattr_getattr_l__mod___stages___3_____1___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____1___conv1_bias = self.getattr_getattr_L__mod___stages___3_____1___conv1_bias
    out_81 = torch.conv2d(out_80, weight_48, getattr_getattr_l__mod___stages___3_____1___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_80 = weight_48 = getattr_getattr_l__mod___stages___3_____1___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___3_____1___act2 = self.getattr_getattr_L__mod___stages___3_____1___act2(out_81);  out_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____1___conv2_weight = self.getattr_getattr_L__mod___stages___3_____1___conv2_weight
    reshape_49 = getattr_getattr_l__mod___stages___3_____1___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____1___conv2_gain = self.getattr_getattr_L__mod___stages___3_____1___conv2_gain
    mul_90 = getattr_getattr_l__mod___stages___3_____1___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____1___conv2_gain = None
    view_49 = mul_90.view(-1);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_49 = torch.nn.functional.batch_norm(reshape_49, None, None, weight = view_49, training = True, momentum = 0.0, eps = 1e-05);  reshape_49 = view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_49 = batch_norm_49.reshape_as(getattr_getattr_l__mod___stages___3_____1___conv2_weight);  batch_norm_49 = getattr_getattr_l__mod___stages___3_____1___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____1___conv2_bias = self.getattr_getattr_L__mod___stages___3_____1___conv2_bias
    out_82 = torch.conv2d(getattr_getattr_l__mod___stages___3_____1___act2, weight_49, getattr_getattr_l__mod___stages___3_____1___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____1___act2 = weight_49 = getattr_getattr_l__mod___stages___3_____1___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___3_____1___act2b = self.getattr_getattr_L__mod___stages___3_____1___act2b(out_82);  out_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____1___conv2b_weight = self.getattr_getattr_L__mod___stages___3_____1___conv2b_weight
    reshape_50 = getattr_getattr_l__mod___stages___3_____1___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____1___conv2b_gain = self.getattr_getattr_L__mod___stages___3_____1___conv2b_gain
    mul_91 = getattr_getattr_l__mod___stages___3_____1___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____1___conv2b_gain = None
    view_50 = mul_91.view(-1);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_50 = torch.nn.functional.batch_norm(reshape_50, None, None, weight = view_50, training = True, momentum = 0.0, eps = 1e-05);  reshape_50 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_50 = batch_norm_50.reshape_as(getattr_getattr_l__mod___stages___3_____1___conv2b_weight);  batch_norm_50 = getattr_getattr_l__mod___stages___3_____1___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____1___conv2b_bias = self.getattr_getattr_L__mod___stages___3_____1___conv2b_bias
    out_83 = torch.conv2d(getattr_getattr_l__mod___stages___3_____1___act2b, weight_50, getattr_getattr_l__mod___stages___3_____1___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____1___act2b = weight_50 = getattr_getattr_l__mod___stages___3_____1___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___3_____1___act3 = self.getattr_getattr_L__mod___stages___3_____1___act3(out_83);  out_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____1___conv3_weight = self.getattr_getattr_L__mod___stages___3_____1___conv3_weight
    reshape_51 = getattr_getattr_l__mod___stages___3_____1___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____1___conv3_gain = self.getattr_getattr_L__mod___stages___3_____1___conv3_gain
    mul_92 = getattr_getattr_l__mod___stages___3_____1___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___3_____1___conv3_gain = None
    view_51 = mul_92.view(-1);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_51 = torch.nn.functional.batch_norm(reshape_51, None, None, weight = view_51, training = True, momentum = 0.0, eps = 1e-05);  reshape_51 = view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_51 = batch_norm_51.reshape_as(getattr_getattr_l__mod___stages___3_____1___conv3_weight);  batch_norm_51 = getattr_getattr_l__mod___stages___3_____1___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____1___conv3_bias = self.getattr_getattr_L__mod___stages___3_____1___conv3_bias
    out_84 = torch.conv2d(getattr_getattr_l__mod___stages___3_____1___act3, weight_51, getattr_getattr_l__mod___stages___3_____1___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___3_____1___act3 = weight_51 = getattr_getattr_l__mod___stages___3_____1___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = out_84.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_41 = self.getattr_getattr_L__mod___stages___3_____1___attn_last_fc1(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___3_____1___attn_last_bn = self.getattr_getattr_L__mod___stages___3_____1___attn_last_bn(x_se_41);  x_se_41 = None
    x_se_42 = self.getattr_getattr_L__mod___stages___3_____1___attn_last_act(getattr_getattr_l__mod___stages___3_____1___attn_last_bn);  getattr_getattr_l__mod___stages___3_____1___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_43 = self.getattr_getattr_L__mod___stages___3_____1___attn_last_fc2(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10 = x_se_43.sigmoid();  x_se_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_93 = out_84 * sigmoid_10;  out_84 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_85 = 2.0 * mul_93;  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_86 = self.getattr_getattr_L__mod___stages___3_____1___drop_path(out_85);  out_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_95 = out_86 * 0.2;  out_86 = None
    shortcut_15 = mul_95 + shortcut_14;  mul_95 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    getattr_getattr_l__mod___stages___3_____2___act1 = self.getattr_getattr_L__mod___stages___3_____2___act1(shortcut_15)
    out_88 = getattr_getattr_l__mod___stages___3_____2___act1 * 0.9622504486493761;  getattr_getattr_l__mod___stages___3_____2___act1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____2___conv1_weight = self.getattr_getattr_L__mod___stages___3_____2___conv1_weight
    reshape_52 = getattr_getattr_l__mod___stages___3_____2___conv1_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____2___conv1_gain = self.getattr_getattr_L__mod___stages___3_____2___conv1_gain
    mul_97 = getattr_getattr_l__mod___stages___3_____2___conv1_gain * 0.04562504637317021;  getattr_getattr_l__mod___stages___3_____2___conv1_gain = None
    view_52 = mul_97.view(-1);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_52 = torch.nn.functional.batch_norm(reshape_52, None, None, weight = view_52, training = True, momentum = 0.0, eps = 1e-05);  reshape_52 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_52 = batch_norm_52.reshape_as(getattr_getattr_l__mod___stages___3_____2___conv1_weight);  batch_norm_52 = getattr_getattr_l__mod___stages___3_____2___conv1_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____2___conv1_bias = self.getattr_getattr_L__mod___stages___3_____2___conv1_bias
    out_89 = torch.conv2d(out_88, weight_52, getattr_getattr_l__mod___stages___3_____2___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  out_88 = weight_52 = getattr_getattr_l__mod___stages___3_____2___conv1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    getattr_getattr_l__mod___stages___3_____2___act2 = self.getattr_getattr_L__mod___stages___3_____2___act2(out_89);  out_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____2___conv2_weight = self.getattr_getattr_L__mod___stages___3_____2___conv2_weight
    reshape_53 = getattr_getattr_l__mod___stages___3_____2___conv2_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____2___conv2_gain = self.getattr_getattr_L__mod___stages___3_____2___conv2_gain
    mul_98 = getattr_getattr_l__mod___stages___3_____2___conv2_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____2___conv2_gain = None
    view_53 = mul_98.view(-1);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_53 = torch.nn.functional.batch_norm(reshape_53, None, None, weight = view_53, training = True, momentum = 0.0, eps = 1e-05);  reshape_53 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_53 = batch_norm_53.reshape_as(getattr_getattr_l__mod___stages___3_____2___conv2_weight);  batch_norm_53 = getattr_getattr_l__mod___stages___3_____2___conv2_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____2___conv2_bias = self.getattr_getattr_L__mod___stages___3_____2___conv2_bias
    out_90 = torch.conv2d(getattr_getattr_l__mod___stages___3_____2___act2, weight_53, getattr_getattr_l__mod___stages___3_____2___conv2_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____2___act2 = weight_53 = getattr_getattr_l__mod___stages___3_____2___conv2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    getattr_getattr_l__mod___stages___3_____2___act2b = self.getattr_getattr_L__mod___stages___3_____2___act2b(out_90);  out_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____2___conv2b_weight = self.getattr_getattr_L__mod___stages___3_____2___conv2b_weight
    reshape_54 = getattr_getattr_l__mod___stages___3_____2___conv2b_weight.reshape(1, 384, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____2___conv2b_gain = self.getattr_getattr_L__mod___stages___3_____2___conv2b_gain
    mul_99 = getattr_getattr_l__mod___stages___3_____2___conv2b_gain * 0.07450538873672485;  getattr_getattr_l__mod___stages___3_____2___conv2b_gain = None
    view_54 = mul_99.view(-1);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_54 = torch.nn.functional.batch_norm(reshape_54, None, None, weight = view_54, training = True, momentum = 0.0, eps = 1e-05);  reshape_54 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_54 = batch_norm_54.reshape_as(getattr_getattr_l__mod___stages___3_____2___conv2b_weight);  batch_norm_54 = getattr_getattr_l__mod___stages___3_____2___conv2b_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____2___conv2b_bias = self.getattr_getattr_L__mod___stages___3_____2___conv2b_bias
    out_91 = torch.conv2d(getattr_getattr_l__mod___stages___3_____2___act2b, weight_54, getattr_getattr_l__mod___stages___3_____2___conv2b_bias, (1, 1), (1, 1), (1, 1), 6);  getattr_getattr_l__mod___stages___3_____2___act2b = weight_54 = getattr_getattr_l__mod___stages___3_____2___conv2b_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    getattr_getattr_l__mod___stages___3_____2___act3 = self.getattr_getattr_L__mod___stages___3_____2___act3(out_91);  out_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    getattr_getattr_l__mod___stages___3_____2___conv3_weight = self.getattr_getattr_L__mod___stages___3_____2___conv3_weight
    reshape_55 = getattr_getattr_l__mod___stages___3_____2___conv3_weight.reshape(1, 1536, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    getattr_getattr_l__mod___stages___3_____2___conv3_gain = self.getattr_getattr_L__mod___stages___3_____2___conv3_gain
    mul_100 = getattr_getattr_l__mod___stages___3_____2___conv3_gain * 0.09125009274634042;  getattr_getattr_l__mod___stages___3_____2___conv3_gain = None
    view_55 = mul_100.view(-1);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_55 = torch.nn.functional.batch_norm(reshape_55, None, None, weight = view_55, training = True, momentum = 0.0, eps = 1e-05);  reshape_55 = view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_55 = batch_norm_55.reshape_as(getattr_getattr_l__mod___stages___3_____2___conv3_weight);  batch_norm_55 = getattr_getattr_l__mod___stages___3_____2___conv3_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    getattr_getattr_l__mod___stages___3_____2___conv3_bias = self.getattr_getattr_L__mod___stages___3_____2___conv3_bias
    out_92 = torch.conv2d(getattr_getattr_l__mod___stages___3_____2___act3, weight_55, getattr_getattr_l__mod___stages___3_____2___conv3_bias, (1, 1), (0, 0), (1, 1), 1);  getattr_getattr_l__mod___stages___3_____2___act3 = weight_55 = getattr_getattr_l__mod___stages___3_____2___conv3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = out_92.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_45 = self.getattr_getattr_L__mod___stages___3_____2___attn_last_fc1(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___3_____2___attn_last_bn = self.getattr_getattr_L__mod___stages___3_____2___attn_last_bn(x_se_45);  x_se_45 = None
    x_se_46 = self.getattr_getattr_L__mod___stages___3_____2___attn_last_act(getattr_getattr_l__mod___stages___3_____2___attn_last_bn);  getattr_getattr_l__mod___stages___3_____2___attn_last_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_47 = self.getattr_getattr_L__mod___stages___3_____2___attn_last_fc2(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11 = x_se_47.sigmoid();  x_se_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_101 = out_92 * sigmoid_11;  out_92 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    out_93 = 2.0 * mul_101;  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:195, code: out = self.drop_path(out)
    out_94 = self.getattr_getattr_L__mod___stages___3_____2___drop_path(out_93);  out_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_103 = out_94 * 0.2;  out_94 = None
    x_1 = mul_103 + shortcut_15;  mul_103 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    l__mod___final_conv_weight = self.L__mod___final_conv_weight
    reshape_56 = l__mod___final_conv_weight.reshape(1, 2304, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    l__mod___final_conv_gain = self.L__mod___final_conv_gain
    mul_104 = l__mod___final_conv_gain * 0.04562504637317021;  l__mod___final_conv_gain = None
    view_56 = mul_104.view(-1);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    batch_norm_56 = torch.nn.functional.batch_norm(reshape_56, None, None, weight = view_56, training = True, momentum = 0.0, eps = 1e-05);  reshape_56 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    weight_56 = batch_norm_56.reshape_as(l__mod___final_conv_weight);  batch_norm_56 = l__mod___final_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    l__mod___final_conv_bias = self.L__mod___final_conv_bias
    x_2 = torch.conv2d(x_1, weight_56, l__mod___final_conv_bias, (1, 1), (0, 0), (1, 1), 1);  x_1 = weight_56 = l__mod___final_conv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    x_4 = self.L__mod___final_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_5 = self.L__mod___head_global_pool_pool(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_7 = self.L__mod___head_global_pool_flatten(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_8 = self.L__mod___head_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_9 = self.L__mod___head_fc(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    pred = self.L__mod___head_flatten(x_9);  x_9 = None
    return (pred,)
    