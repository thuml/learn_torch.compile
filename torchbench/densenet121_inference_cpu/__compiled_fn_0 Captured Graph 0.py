from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    l__mod___features_conv0 = self.L__mod___features_conv0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___features_norm0 = self.L__mod___features_norm0(l__mod___features_conv0);  l__mod___features_conv0 = None
    l__mod___features_relu0 = self.L__mod___features_relu0(l__mod___features_norm0);  l__mod___features_norm0 = None
    l__mod___features_pool0 = self.L__mod___features_pool0(l__mod___features_relu0);  l__mod___features_relu0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features = torch.cat([l__mod___features_pool0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer1_norm1 = self.L__mod___features_denseblock1_denselayer1_norm1(concated_features);  concated_features = None
    l__mod___features_denseblock1_denselayer1_relu1 = self.L__mod___features_denseblock1_denselayer1_relu1(l__mod___features_denseblock1_denselayer1_norm1);  l__mod___features_denseblock1_denselayer1_norm1 = None
    bottleneck_output = self.L__mod___features_denseblock1_denselayer1_conv1(l__mod___features_denseblock1_denselayer1_relu1);  l__mod___features_denseblock1_denselayer1_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer1_norm2 = self.L__mod___features_denseblock1_denselayer1_norm2(bottleneck_output);  bottleneck_output = None
    l__mod___features_denseblock1_denselayer1_relu2 = self.L__mod___features_denseblock1_denselayer1_relu2(l__mod___features_denseblock1_denselayer1_norm2);  l__mod___features_denseblock1_denselayer1_norm2 = None
    new_features = self.L__mod___features_denseblock1_denselayer1_conv2(l__mod___features_denseblock1_denselayer1_relu2);  l__mod___features_denseblock1_denselayer1_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_1 = torch.cat([l__mod___features_pool0, new_features], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer2_norm1 = self.L__mod___features_denseblock1_denselayer2_norm1(concated_features_1);  concated_features_1 = None
    l__mod___features_denseblock1_denselayer2_relu1 = self.L__mod___features_denseblock1_denselayer2_relu1(l__mod___features_denseblock1_denselayer2_norm1);  l__mod___features_denseblock1_denselayer2_norm1 = None
    bottleneck_output_2 = self.L__mod___features_denseblock1_denselayer2_conv1(l__mod___features_denseblock1_denselayer2_relu1);  l__mod___features_denseblock1_denselayer2_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer2_norm2 = self.L__mod___features_denseblock1_denselayer2_norm2(bottleneck_output_2);  bottleneck_output_2 = None
    l__mod___features_denseblock1_denselayer2_relu2 = self.L__mod___features_denseblock1_denselayer2_relu2(l__mod___features_denseblock1_denselayer2_norm2);  l__mod___features_denseblock1_denselayer2_norm2 = None
    new_features_2 = self.L__mod___features_denseblock1_denselayer2_conv2(l__mod___features_denseblock1_denselayer2_relu2);  l__mod___features_denseblock1_denselayer2_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_2 = torch.cat([l__mod___features_pool0, new_features, new_features_2], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer3_norm1 = self.L__mod___features_denseblock1_denselayer3_norm1(concated_features_2);  concated_features_2 = None
    l__mod___features_denseblock1_denselayer3_relu1 = self.L__mod___features_denseblock1_denselayer3_relu1(l__mod___features_denseblock1_denselayer3_norm1);  l__mod___features_denseblock1_denselayer3_norm1 = None
    bottleneck_output_4 = self.L__mod___features_denseblock1_denselayer3_conv1(l__mod___features_denseblock1_denselayer3_relu1);  l__mod___features_denseblock1_denselayer3_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer3_norm2 = self.L__mod___features_denseblock1_denselayer3_norm2(bottleneck_output_4);  bottleneck_output_4 = None
    l__mod___features_denseblock1_denselayer3_relu2 = self.L__mod___features_denseblock1_denselayer3_relu2(l__mod___features_denseblock1_denselayer3_norm2);  l__mod___features_denseblock1_denselayer3_norm2 = None
    new_features_4 = self.L__mod___features_denseblock1_denselayer3_conv2(l__mod___features_denseblock1_denselayer3_relu2);  l__mod___features_denseblock1_denselayer3_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_3 = torch.cat([l__mod___features_pool0, new_features, new_features_2, new_features_4], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer4_norm1 = self.L__mod___features_denseblock1_denselayer4_norm1(concated_features_3);  concated_features_3 = None
    l__mod___features_denseblock1_denselayer4_relu1 = self.L__mod___features_denseblock1_denselayer4_relu1(l__mod___features_denseblock1_denselayer4_norm1);  l__mod___features_denseblock1_denselayer4_norm1 = None
    bottleneck_output_6 = self.L__mod___features_denseblock1_denselayer4_conv1(l__mod___features_denseblock1_denselayer4_relu1);  l__mod___features_denseblock1_denselayer4_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer4_norm2 = self.L__mod___features_denseblock1_denselayer4_norm2(bottleneck_output_6);  bottleneck_output_6 = None
    l__mod___features_denseblock1_denselayer4_relu2 = self.L__mod___features_denseblock1_denselayer4_relu2(l__mod___features_denseblock1_denselayer4_norm2);  l__mod___features_denseblock1_denselayer4_norm2 = None
    new_features_6 = self.L__mod___features_denseblock1_denselayer4_conv2(l__mod___features_denseblock1_denselayer4_relu2);  l__mod___features_denseblock1_denselayer4_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_4 = torch.cat([l__mod___features_pool0, new_features, new_features_2, new_features_4, new_features_6], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer5_norm1 = self.L__mod___features_denseblock1_denselayer5_norm1(concated_features_4);  concated_features_4 = None
    l__mod___features_denseblock1_denselayer5_relu1 = self.L__mod___features_denseblock1_denselayer5_relu1(l__mod___features_denseblock1_denselayer5_norm1);  l__mod___features_denseblock1_denselayer5_norm1 = None
    bottleneck_output_8 = self.L__mod___features_denseblock1_denselayer5_conv1(l__mod___features_denseblock1_denselayer5_relu1);  l__mod___features_denseblock1_denselayer5_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer5_norm2 = self.L__mod___features_denseblock1_denselayer5_norm2(bottleneck_output_8);  bottleneck_output_8 = None
    l__mod___features_denseblock1_denselayer5_relu2 = self.L__mod___features_denseblock1_denselayer5_relu2(l__mod___features_denseblock1_denselayer5_norm2);  l__mod___features_denseblock1_denselayer5_norm2 = None
    new_features_8 = self.L__mod___features_denseblock1_denselayer5_conv2(l__mod___features_denseblock1_denselayer5_relu2);  l__mod___features_denseblock1_denselayer5_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_5 = torch.cat([l__mod___features_pool0, new_features, new_features_2, new_features_4, new_features_6, new_features_8], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock1_denselayer6_norm1 = self.L__mod___features_denseblock1_denselayer6_norm1(concated_features_5);  concated_features_5 = None
    l__mod___features_denseblock1_denselayer6_relu1 = self.L__mod___features_denseblock1_denselayer6_relu1(l__mod___features_denseblock1_denselayer6_norm1);  l__mod___features_denseblock1_denselayer6_norm1 = None
    bottleneck_output_10 = self.L__mod___features_denseblock1_denselayer6_conv1(l__mod___features_denseblock1_denselayer6_relu1);  l__mod___features_denseblock1_denselayer6_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock1_denselayer6_norm2 = self.L__mod___features_denseblock1_denselayer6_norm2(bottleneck_output_10);  bottleneck_output_10 = None
    l__mod___features_denseblock1_denselayer6_relu2 = self.L__mod___features_denseblock1_denselayer6_relu2(l__mod___features_denseblock1_denselayer6_norm2);  l__mod___features_denseblock1_denselayer6_norm2 = None
    new_features_10 = self.L__mod___features_denseblock1_denselayer6_conv2(l__mod___features_denseblock1_denselayer6_relu2);  l__mod___features_denseblock1_denselayer6_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_6 = torch.cat([l__mod___features_pool0, new_features, new_features_2, new_features_4, new_features_6, new_features_8, new_features_10], 1);  l__mod___features_pool0 = new_features = new_features_2 = new_features_4 = new_features_6 = new_features_8 = new_features_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    l__mod___features_transition1_norm = self.L__mod___features_transition1_norm(cat_6);  cat_6 = None
    l__mod___features_transition1_relu = self.L__mod___features_transition1_relu(l__mod___features_transition1_norm);  l__mod___features_transition1_norm = None
    l__mod___features_transition1_conv = self.L__mod___features_transition1_conv(l__mod___features_transition1_relu);  l__mod___features_transition1_relu = None
    l__mod___features_transition1_pool = self.L__mod___features_transition1_pool(l__mod___features_transition1_conv);  l__mod___features_transition1_conv = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_6 = torch.cat([l__mod___features_transition1_pool], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer1_norm1 = self.L__mod___features_denseblock2_denselayer1_norm1(concated_features_6);  concated_features_6 = None
    l__mod___features_denseblock2_denselayer1_relu1 = self.L__mod___features_denseblock2_denselayer1_relu1(l__mod___features_denseblock2_denselayer1_norm1);  l__mod___features_denseblock2_denselayer1_norm1 = None
    bottleneck_output_12 = self.L__mod___features_denseblock2_denselayer1_conv1(l__mod___features_denseblock2_denselayer1_relu1);  l__mod___features_denseblock2_denselayer1_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer1_norm2 = self.L__mod___features_denseblock2_denselayer1_norm2(bottleneck_output_12);  bottleneck_output_12 = None
    l__mod___features_denseblock2_denselayer1_relu2 = self.L__mod___features_denseblock2_denselayer1_relu2(l__mod___features_denseblock2_denselayer1_norm2);  l__mod___features_denseblock2_denselayer1_norm2 = None
    new_features_12 = self.L__mod___features_denseblock2_denselayer1_conv2(l__mod___features_denseblock2_denselayer1_relu2);  l__mod___features_denseblock2_denselayer1_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_7 = torch.cat([l__mod___features_transition1_pool, new_features_12], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer2_norm1 = self.L__mod___features_denseblock2_denselayer2_norm1(concated_features_7);  concated_features_7 = None
    l__mod___features_denseblock2_denselayer2_relu1 = self.L__mod___features_denseblock2_denselayer2_relu1(l__mod___features_denseblock2_denselayer2_norm1);  l__mod___features_denseblock2_denselayer2_norm1 = None
    bottleneck_output_14 = self.L__mod___features_denseblock2_denselayer2_conv1(l__mod___features_denseblock2_denselayer2_relu1);  l__mod___features_denseblock2_denselayer2_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer2_norm2 = self.L__mod___features_denseblock2_denselayer2_norm2(bottleneck_output_14);  bottleneck_output_14 = None
    l__mod___features_denseblock2_denselayer2_relu2 = self.L__mod___features_denseblock2_denselayer2_relu2(l__mod___features_denseblock2_denselayer2_norm2);  l__mod___features_denseblock2_denselayer2_norm2 = None
    new_features_14 = self.L__mod___features_denseblock2_denselayer2_conv2(l__mod___features_denseblock2_denselayer2_relu2);  l__mod___features_denseblock2_denselayer2_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_8 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer3_norm1 = self.L__mod___features_denseblock2_denselayer3_norm1(concated_features_8);  concated_features_8 = None
    l__mod___features_denseblock2_denselayer3_relu1 = self.L__mod___features_denseblock2_denselayer3_relu1(l__mod___features_denseblock2_denselayer3_norm1);  l__mod___features_denseblock2_denselayer3_norm1 = None
    bottleneck_output_16 = self.L__mod___features_denseblock2_denselayer3_conv1(l__mod___features_denseblock2_denselayer3_relu1);  l__mod___features_denseblock2_denselayer3_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer3_norm2 = self.L__mod___features_denseblock2_denselayer3_norm2(bottleneck_output_16);  bottleneck_output_16 = None
    l__mod___features_denseblock2_denselayer3_relu2 = self.L__mod___features_denseblock2_denselayer3_relu2(l__mod___features_denseblock2_denselayer3_norm2);  l__mod___features_denseblock2_denselayer3_norm2 = None
    new_features_16 = self.L__mod___features_denseblock2_denselayer3_conv2(l__mod___features_denseblock2_denselayer3_relu2);  l__mod___features_denseblock2_denselayer3_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_9 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer4_norm1 = self.L__mod___features_denseblock2_denselayer4_norm1(concated_features_9);  concated_features_9 = None
    l__mod___features_denseblock2_denselayer4_relu1 = self.L__mod___features_denseblock2_denselayer4_relu1(l__mod___features_denseblock2_denselayer4_norm1);  l__mod___features_denseblock2_denselayer4_norm1 = None
    bottleneck_output_18 = self.L__mod___features_denseblock2_denselayer4_conv1(l__mod___features_denseblock2_denselayer4_relu1);  l__mod___features_denseblock2_denselayer4_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer4_norm2 = self.L__mod___features_denseblock2_denselayer4_norm2(bottleneck_output_18);  bottleneck_output_18 = None
    l__mod___features_denseblock2_denselayer4_relu2 = self.L__mod___features_denseblock2_denselayer4_relu2(l__mod___features_denseblock2_denselayer4_norm2);  l__mod___features_denseblock2_denselayer4_norm2 = None
    new_features_18 = self.L__mod___features_denseblock2_denselayer4_conv2(l__mod___features_denseblock2_denselayer4_relu2);  l__mod___features_denseblock2_denselayer4_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_10 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer5_norm1 = self.L__mod___features_denseblock2_denselayer5_norm1(concated_features_10);  concated_features_10 = None
    l__mod___features_denseblock2_denselayer5_relu1 = self.L__mod___features_denseblock2_denselayer5_relu1(l__mod___features_denseblock2_denselayer5_norm1);  l__mod___features_denseblock2_denselayer5_norm1 = None
    bottleneck_output_20 = self.L__mod___features_denseblock2_denselayer5_conv1(l__mod___features_denseblock2_denselayer5_relu1);  l__mod___features_denseblock2_denselayer5_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer5_norm2 = self.L__mod___features_denseblock2_denselayer5_norm2(bottleneck_output_20);  bottleneck_output_20 = None
    l__mod___features_denseblock2_denselayer5_relu2 = self.L__mod___features_denseblock2_denselayer5_relu2(l__mod___features_denseblock2_denselayer5_norm2);  l__mod___features_denseblock2_denselayer5_norm2 = None
    new_features_20 = self.L__mod___features_denseblock2_denselayer5_conv2(l__mod___features_denseblock2_denselayer5_relu2);  l__mod___features_denseblock2_denselayer5_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_11 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer6_norm1 = self.L__mod___features_denseblock2_denselayer6_norm1(concated_features_11);  concated_features_11 = None
    l__mod___features_denseblock2_denselayer6_relu1 = self.L__mod___features_denseblock2_denselayer6_relu1(l__mod___features_denseblock2_denselayer6_norm1);  l__mod___features_denseblock2_denselayer6_norm1 = None
    bottleneck_output_22 = self.L__mod___features_denseblock2_denselayer6_conv1(l__mod___features_denseblock2_denselayer6_relu1);  l__mod___features_denseblock2_denselayer6_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer6_norm2 = self.L__mod___features_denseblock2_denselayer6_norm2(bottleneck_output_22);  bottleneck_output_22 = None
    l__mod___features_denseblock2_denselayer6_relu2 = self.L__mod___features_denseblock2_denselayer6_relu2(l__mod___features_denseblock2_denselayer6_norm2);  l__mod___features_denseblock2_denselayer6_norm2 = None
    new_features_22 = self.L__mod___features_denseblock2_denselayer6_conv2(l__mod___features_denseblock2_denselayer6_relu2);  l__mod___features_denseblock2_denselayer6_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_12 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer7_norm1 = self.L__mod___features_denseblock2_denselayer7_norm1(concated_features_12);  concated_features_12 = None
    l__mod___features_denseblock2_denselayer7_relu1 = self.L__mod___features_denseblock2_denselayer7_relu1(l__mod___features_denseblock2_denselayer7_norm1);  l__mod___features_denseblock2_denselayer7_norm1 = None
    bottleneck_output_24 = self.L__mod___features_denseblock2_denselayer7_conv1(l__mod___features_denseblock2_denselayer7_relu1);  l__mod___features_denseblock2_denselayer7_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer7_norm2 = self.L__mod___features_denseblock2_denselayer7_norm2(bottleneck_output_24);  bottleneck_output_24 = None
    l__mod___features_denseblock2_denselayer7_relu2 = self.L__mod___features_denseblock2_denselayer7_relu2(l__mod___features_denseblock2_denselayer7_norm2);  l__mod___features_denseblock2_denselayer7_norm2 = None
    new_features_24 = self.L__mod___features_denseblock2_denselayer7_conv2(l__mod___features_denseblock2_denselayer7_relu2);  l__mod___features_denseblock2_denselayer7_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_13 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer8_norm1 = self.L__mod___features_denseblock2_denselayer8_norm1(concated_features_13);  concated_features_13 = None
    l__mod___features_denseblock2_denselayer8_relu1 = self.L__mod___features_denseblock2_denselayer8_relu1(l__mod___features_denseblock2_denselayer8_norm1);  l__mod___features_denseblock2_denselayer8_norm1 = None
    bottleneck_output_26 = self.L__mod___features_denseblock2_denselayer8_conv1(l__mod___features_denseblock2_denselayer8_relu1);  l__mod___features_denseblock2_denselayer8_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer8_norm2 = self.L__mod___features_denseblock2_denselayer8_norm2(bottleneck_output_26);  bottleneck_output_26 = None
    l__mod___features_denseblock2_denselayer8_relu2 = self.L__mod___features_denseblock2_denselayer8_relu2(l__mod___features_denseblock2_denselayer8_norm2);  l__mod___features_denseblock2_denselayer8_norm2 = None
    new_features_26 = self.L__mod___features_denseblock2_denselayer8_conv2(l__mod___features_denseblock2_denselayer8_relu2);  l__mod___features_denseblock2_denselayer8_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_14 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24, new_features_26], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer9_norm1 = self.L__mod___features_denseblock2_denselayer9_norm1(concated_features_14);  concated_features_14 = None
    l__mod___features_denseblock2_denselayer9_relu1 = self.L__mod___features_denseblock2_denselayer9_relu1(l__mod___features_denseblock2_denselayer9_norm1);  l__mod___features_denseblock2_denselayer9_norm1 = None
    bottleneck_output_28 = self.L__mod___features_denseblock2_denselayer9_conv1(l__mod___features_denseblock2_denselayer9_relu1);  l__mod___features_denseblock2_denselayer9_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer9_norm2 = self.L__mod___features_denseblock2_denselayer9_norm2(bottleneck_output_28);  bottleneck_output_28 = None
    l__mod___features_denseblock2_denselayer9_relu2 = self.L__mod___features_denseblock2_denselayer9_relu2(l__mod___features_denseblock2_denselayer9_norm2);  l__mod___features_denseblock2_denselayer9_norm2 = None
    new_features_28 = self.L__mod___features_denseblock2_denselayer9_conv2(l__mod___features_denseblock2_denselayer9_relu2);  l__mod___features_denseblock2_denselayer9_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_15 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24, new_features_26, new_features_28], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer10_norm1 = self.L__mod___features_denseblock2_denselayer10_norm1(concated_features_15);  concated_features_15 = None
    l__mod___features_denseblock2_denselayer10_relu1 = self.L__mod___features_denseblock2_denselayer10_relu1(l__mod___features_denseblock2_denselayer10_norm1);  l__mod___features_denseblock2_denselayer10_norm1 = None
    bottleneck_output_30 = self.L__mod___features_denseblock2_denselayer10_conv1(l__mod___features_denseblock2_denselayer10_relu1);  l__mod___features_denseblock2_denselayer10_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer10_norm2 = self.L__mod___features_denseblock2_denselayer10_norm2(bottleneck_output_30);  bottleneck_output_30 = None
    l__mod___features_denseblock2_denselayer10_relu2 = self.L__mod___features_denseblock2_denselayer10_relu2(l__mod___features_denseblock2_denselayer10_norm2);  l__mod___features_denseblock2_denselayer10_norm2 = None
    new_features_30 = self.L__mod___features_denseblock2_denselayer10_conv2(l__mod___features_denseblock2_denselayer10_relu2);  l__mod___features_denseblock2_denselayer10_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_16 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24, new_features_26, new_features_28, new_features_30], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer11_norm1 = self.L__mod___features_denseblock2_denselayer11_norm1(concated_features_16);  concated_features_16 = None
    l__mod___features_denseblock2_denselayer11_relu1 = self.L__mod___features_denseblock2_denselayer11_relu1(l__mod___features_denseblock2_denselayer11_norm1);  l__mod___features_denseblock2_denselayer11_norm1 = None
    bottleneck_output_32 = self.L__mod___features_denseblock2_denselayer11_conv1(l__mod___features_denseblock2_denselayer11_relu1);  l__mod___features_denseblock2_denselayer11_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer11_norm2 = self.L__mod___features_denseblock2_denselayer11_norm2(bottleneck_output_32);  bottleneck_output_32 = None
    l__mod___features_denseblock2_denselayer11_relu2 = self.L__mod___features_denseblock2_denselayer11_relu2(l__mod___features_denseblock2_denselayer11_norm2);  l__mod___features_denseblock2_denselayer11_norm2 = None
    new_features_32 = self.L__mod___features_denseblock2_denselayer11_conv2(l__mod___features_denseblock2_denselayer11_relu2);  l__mod___features_denseblock2_denselayer11_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_17 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24, new_features_26, new_features_28, new_features_30, new_features_32], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock2_denselayer12_norm1 = self.L__mod___features_denseblock2_denselayer12_norm1(concated_features_17);  concated_features_17 = None
    l__mod___features_denseblock2_denselayer12_relu1 = self.L__mod___features_denseblock2_denselayer12_relu1(l__mod___features_denseblock2_denselayer12_norm1);  l__mod___features_denseblock2_denselayer12_norm1 = None
    bottleneck_output_34 = self.L__mod___features_denseblock2_denselayer12_conv1(l__mod___features_denseblock2_denselayer12_relu1);  l__mod___features_denseblock2_denselayer12_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock2_denselayer12_norm2 = self.L__mod___features_denseblock2_denselayer12_norm2(bottleneck_output_34);  bottleneck_output_34 = None
    l__mod___features_denseblock2_denselayer12_relu2 = self.L__mod___features_denseblock2_denselayer12_relu2(l__mod___features_denseblock2_denselayer12_norm2);  l__mod___features_denseblock2_denselayer12_norm2 = None
    new_features_34 = self.L__mod___features_denseblock2_denselayer12_conv2(l__mod___features_denseblock2_denselayer12_relu2);  l__mod___features_denseblock2_denselayer12_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_19 = torch.cat([l__mod___features_transition1_pool, new_features_12, new_features_14, new_features_16, new_features_18, new_features_20, new_features_22, new_features_24, new_features_26, new_features_28, new_features_30, new_features_32, new_features_34], 1);  l__mod___features_transition1_pool = new_features_12 = new_features_14 = new_features_16 = new_features_18 = new_features_20 = new_features_22 = new_features_24 = new_features_26 = new_features_28 = new_features_30 = new_features_32 = new_features_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    l__mod___features_transition2_norm = self.L__mod___features_transition2_norm(cat_19);  cat_19 = None
    l__mod___features_transition2_relu = self.L__mod___features_transition2_relu(l__mod___features_transition2_norm);  l__mod___features_transition2_norm = None
    l__mod___features_transition2_conv = self.L__mod___features_transition2_conv(l__mod___features_transition2_relu);  l__mod___features_transition2_relu = None
    l__mod___features_transition2_pool = self.L__mod___features_transition2_pool(l__mod___features_transition2_conv);  l__mod___features_transition2_conv = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_18 = torch.cat([l__mod___features_transition2_pool], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer1_norm1 = self.L__mod___features_denseblock3_denselayer1_norm1(concated_features_18);  concated_features_18 = None
    l__mod___features_denseblock3_denselayer1_relu1 = self.L__mod___features_denseblock3_denselayer1_relu1(l__mod___features_denseblock3_denselayer1_norm1);  l__mod___features_denseblock3_denselayer1_norm1 = None
    bottleneck_output_36 = self.L__mod___features_denseblock3_denselayer1_conv1(l__mod___features_denseblock3_denselayer1_relu1);  l__mod___features_denseblock3_denselayer1_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer1_norm2 = self.L__mod___features_denseblock3_denselayer1_norm2(bottleneck_output_36);  bottleneck_output_36 = None
    l__mod___features_denseblock3_denselayer1_relu2 = self.L__mod___features_denseblock3_denselayer1_relu2(l__mod___features_denseblock3_denselayer1_norm2);  l__mod___features_denseblock3_denselayer1_norm2 = None
    new_features_36 = self.L__mod___features_denseblock3_denselayer1_conv2(l__mod___features_denseblock3_denselayer1_relu2);  l__mod___features_denseblock3_denselayer1_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_19 = torch.cat([l__mod___features_transition2_pool, new_features_36], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer2_norm1 = self.L__mod___features_denseblock3_denselayer2_norm1(concated_features_19);  concated_features_19 = None
    l__mod___features_denseblock3_denselayer2_relu1 = self.L__mod___features_denseblock3_denselayer2_relu1(l__mod___features_denseblock3_denselayer2_norm1);  l__mod___features_denseblock3_denselayer2_norm1 = None
    bottleneck_output_38 = self.L__mod___features_denseblock3_denselayer2_conv1(l__mod___features_denseblock3_denselayer2_relu1);  l__mod___features_denseblock3_denselayer2_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer2_norm2 = self.L__mod___features_denseblock3_denselayer2_norm2(bottleneck_output_38);  bottleneck_output_38 = None
    l__mod___features_denseblock3_denselayer2_relu2 = self.L__mod___features_denseblock3_denselayer2_relu2(l__mod___features_denseblock3_denselayer2_norm2);  l__mod___features_denseblock3_denselayer2_norm2 = None
    new_features_38 = self.L__mod___features_denseblock3_denselayer2_conv2(l__mod___features_denseblock3_denselayer2_relu2);  l__mod___features_denseblock3_denselayer2_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_20 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer3_norm1 = self.L__mod___features_denseblock3_denselayer3_norm1(concated_features_20);  concated_features_20 = None
    l__mod___features_denseblock3_denselayer3_relu1 = self.L__mod___features_denseblock3_denselayer3_relu1(l__mod___features_denseblock3_denselayer3_norm1);  l__mod___features_denseblock3_denselayer3_norm1 = None
    bottleneck_output_40 = self.L__mod___features_denseblock3_denselayer3_conv1(l__mod___features_denseblock3_denselayer3_relu1);  l__mod___features_denseblock3_denselayer3_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer3_norm2 = self.L__mod___features_denseblock3_denselayer3_norm2(bottleneck_output_40);  bottleneck_output_40 = None
    l__mod___features_denseblock3_denselayer3_relu2 = self.L__mod___features_denseblock3_denselayer3_relu2(l__mod___features_denseblock3_denselayer3_norm2);  l__mod___features_denseblock3_denselayer3_norm2 = None
    new_features_40 = self.L__mod___features_denseblock3_denselayer3_conv2(l__mod___features_denseblock3_denselayer3_relu2);  l__mod___features_denseblock3_denselayer3_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_21 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer4_norm1 = self.L__mod___features_denseblock3_denselayer4_norm1(concated_features_21);  concated_features_21 = None
    l__mod___features_denseblock3_denselayer4_relu1 = self.L__mod___features_denseblock3_denselayer4_relu1(l__mod___features_denseblock3_denselayer4_norm1);  l__mod___features_denseblock3_denselayer4_norm1 = None
    bottleneck_output_42 = self.L__mod___features_denseblock3_denselayer4_conv1(l__mod___features_denseblock3_denselayer4_relu1);  l__mod___features_denseblock3_denselayer4_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer4_norm2 = self.L__mod___features_denseblock3_denselayer4_norm2(bottleneck_output_42);  bottleneck_output_42 = None
    l__mod___features_denseblock3_denselayer4_relu2 = self.L__mod___features_denseblock3_denselayer4_relu2(l__mod___features_denseblock3_denselayer4_norm2);  l__mod___features_denseblock3_denselayer4_norm2 = None
    new_features_42 = self.L__mod___features_denseblock3_denselayer4_conv2(l__mod___features_denseblock3_denselayer4_relu2);  l__mod___features_denseblock3_denselayer4_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_22 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer5_norm1 = self.L__mod___features_denseblock3_denselayer5_norm1(concated_features_22);  concated_features_22 = None
    l__mod___features_denseblock3_denselayer5_relu1 = self.L__mod___features_denseblock3_denselayer5_relu1(l__mod___features_denseblock3_denselayer5_norm1);  l__mod___features_denseblock3_denselayer5_norm1 = None
    bottleneck_output_44 = self.L__mod___features_denseblock3_denselayer5_conv1(l__mod___features_denseblock3_denselayer5_relu1);  l__mod___features_denseblock3_denselayer5_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer5_norm2 = self.L__mod___features_denseblock3_denselayer5_norm2(bottleneck_output_44);  bottleneck_output_44 = None
    l__mod___features_denseblock3_denselayer5_relu2 = self.L__mod___features_denseblock3_denselayer5_relu2(l__mod___features_denseblock3_denselayer5_norm2);  l__mod___features_denseblock3_denselayer5_norm2 = None
    new_features_44 = self.L__mod___features_denseblock3_denselayer5_conv2(l__mod___features_denseblock3_denselayer5_relu2);  l__mod___features_denseblock3_denselayer5_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_23 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer6_norm1 = self.L__mod___features_denseblock3_denselayer6_norm1(concated_features_23);  concated_features_23 = None
    l__mod___features_denseblock3_denselayer6_relu1 = self.L__mod___features_denseblock3_denselayer6_relu1(l__mod___features_denseblock3_denselayer6_norm1);  l__mod___features_denseblock3_denselayer6_norm1 = None
    bottleneck_output_46 = self.L__mod___features_denseblock3_denselayer6_conv1(l__mod___features_denseblock3_denselayer6_relu1);  l__mod___features_denseblock3_denselayer6_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer6_norm2 = self.L__mod___features_denseblock3_denselayer6_norm2(bottleneck_output_46);  bottleneck_output_46 = None
    l__mod___features_denseblock3_denselayer6_relu2 = self.L__mod___features_denseblock3_denselayer6_relu2(l__mod___features_denseblock3_denselayer6_norm2);  l__mod___features_denseblock3_denselayer6_norm2 = None
    new_features_46 = self.L__mod___features_denseblock3_denselayer6_conv2(l__mod___features_denseblock3_denselayer6_relu2);  l__mod___features_denseblock3_denselayer6_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_24 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer7_norm1 = self.L__mod___features_denseblock3_denselayer7_norm1(concated_features_24);  concated_features_24 = None
    l__mod___features_denseblock3_denselayer7_relu1 = self.L__mod___features_denseblock3_denselayer7_relu1(l__mod___features_denseblock3_denselayer7_norm1);  l__mod___features_denseblock3_denselayer7_norm1 = None
    bottleneck_output_48 = self.L__mod___features_denseblock3_denselayer7_conv1(l__mod___features_denseblock3_denselayer7_relu1);  l__mod___features_denseblock3_denselayer7_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer7_norm2 = self.L__mod___features_denseblock3_denselayer7_norm2(bottleneck_output_48);  bottleneck_output_48 = None
    l__mod___features_denseblock3_denselayer7_relu2 = self.L__mod___features_denseblock3_denselayer7_relu2(l__mod___features_denseblock3_denselayer7_norm2);  l__mod___features_denseblock3_denselayer7_norm2 = None
    new_features_48 = self.L__mod___features_denseblock3_denselayer7_conv2(l__mod___features_denseblock3_denselayer7_relu2);  l__mod___features_denseblock3_denselayer7_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_25 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer8_norm1 = self.L__mod___features_denseblock3_denselayer8_norm1(concated_features_25);  concated_features_25 = None
    l__mod___features_denseblock3_denselayer8_relu1 = self.L__mod___features_denseblock3_denselayer8_relu1(l__mod___features_denseblock3_denselayer8_norm1);  l__mod___features_denseblock3_denselayer8_norm1 = None
    bottleneck_output_50 = self.L__mod___features_denseblock3_denselayer8_conv1(l__mod___features_denseblock3_denselayer8_relu1);  l__mod___features_denseblock3_denselayer8_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer8_norm2 = self.L__mod___features_denseblock3_denselayer8_norm2(bottleneck_output_50);  bottleneck_output_50 = None
    l__mod___features_denseblock3_denselayer8_relu2 = self.L__mod___features_denseblock3_denselayer8_relu2(l__mod___features_denseblock3_denselayer8_norm2);  l__mod___features_denseblock3_denselayer8_norm2 = None
    new_features_50 = self.L__mod___features_denseblock3_denselayer8_conv2(l__mod___features_denseblock3_denselayer8_relu2);  l__mod___features_denseblock3_denselayer8_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_26 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer9_norm1 = self.L__mod___features_denseblock3_denselayer9_norm1(concated_features_26);  concated_features_26 = None
    l__mod___features_denseblock3_denselayer9_relu1 = self.L__mod___features_denseblock3_denselayer9_relu1(l__mod___features_denseblock3_denselayer9_norm1);  l__mod___features_denseblock3_denselayer9_norm1 = None
    bottleneck_output_52 = self.L__mod___features_denseblock3_denselayer9_conv1(l__mod___features_denseblock3_denselayer9_relu1);  l__mod___features_denseblock3_denselayer9_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer9_norm2 = self.L__mod___features_denseblock3_denselayer9_norm2(bottleneck_output_52);  bottleneck_output_52 = None
    l__mod___features_denseblock3_denselayer9_relu2 = self.L__mod___features_denseblock3_denselayer9_relu2(l__mod___features_denseblock3_denselayer9_norm2);  l__mod___features_denseblock3_denselayer9_norm2 = None
    new_features_52 = self.L__mod___features_denseblock3_denselayer9_conv2(l__mod___features_denseblock3_denselayer9_relu2);  l__mod___features_denseblock3_denselayer9_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_27 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer10_norm1 = self.L__mod___features_denseblock3_denselayer10_norm1(concated_features_27);  concated_features_27 = None
    l__mod___features_denseblock3_denselayer10_relu1 = self.L__mod___features_denseblock3_denselayer10_relu1(l__mod___features_denseblock3_denselayer10_norm1);  l__mod___features_denseblock3_denselayer10_norm1 = None
    bottleneck_output_54 = self.L__mod___features_denseblock3_denselayer10_conv1(l__mod___features_denseblock3_denselayer10_relu1);  l__mod___features_denseblock3_denselayer10_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer10_norm2 = self.L__mod___features_denseblock3_denselayer10_norm2(bottleneck_output_54);  bottleneck_output_54 = None
    l__mod___features_denseblock3_denselayer10_relu2 = self.L__mod___features_denseblock3_denselayer10_relu2(l__mod___features_denseblock3_denselayer10_norm2);  l__mod___features_denseblock3_denselayer10_norm2 = None
    new_features_54 = self.L__mod___features_denseblock3_denselayer10_conv2(l__mod___features_denseblock3_denselayer10_relu2);  l__mod___features_denseblock3_denselayer10_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_28 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer11_norm1 = self.L__mod___features_denseblock3_denselayer11_norm1(concated_features_28);  concated_features_28 = None
    l__mod___features_denseblock3_denselayer11_relu1 = self.L__mod___features_denseblock3_denselayer11_relu1(l__mod___features_denseblock3_denselayer11_norm1);  l__mod___features_denseblock3_denselayer11_norm1 = None
    bottleneck_output_56 = self.L__mod___features_denseblock3_denselayer11_conv1(l__mod___features_denseblock3_denselayer11_relu1);  l__mod___features_denseblock3_denselayer11_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer11_norm2 = self.L__mod___features_denseblock3_denselayer11_norm2(bottleneck_output_56);  bottleneck_output_56 = None
    l__mod___features_denseblock3_denselayer11_relu2 = self.L__mod___features_denseblock3_denselayer11_relu2(l__mod___features_denseblock3_denselayer11_norm2);  l__mod___features_denseblock3_denselayer11_norm2 = None
    new_features_56 = self.L__mod___features_denseblock3_denselayer11_conv2(l__mod___features_denseblock3_denselayer11_relu2);  l__mod___features_denseblock3_denselayer11_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_29 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer12_norm1 = self.L__mod___features_denseblock3_denselayer12_norm1(concated_features_29);  concated_features_29 = None
    l__mod___features_denseblock3_denselayer12_relu1 = self.L__mod___features_denseblock3_denselayer12_relu1(l__mod___features_denseblock3_denselayer12_norm1);  l__mod___features_denseblock3_denselayer12_norm1 = None
    bottleneck_output_58 = self.L__mod___features_denseblock3_denselayer12_conv1(l__mod___features_denseblock3_denselayer12_relu1);  l__mod___features_denseblock3_denselayer12_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer12_norm2 = self.L__mod___features_denseblock3_denselayer12_norm2(bottleneck_output_58);  bottleneck_output_58 = None
    l__mod___features_denseblock3_denselayer12_relu2 = self.L__mod___features_denseblock3_denselayer12_relu2(l__mod___features_denseblock3_denselayer12_norm2);  l__mod___features_denseblock3_denselayer12_norm2 = None
    new_features_58 = self.L__mod___features_denseblock3_denselayer12_conv2(l__mod___features_denseblock3_denselayer12_relu2);  l__mod___features_denseblock3_denselayer12_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_30 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer13_norm1 = self.L__mod___features_denseblock3_denselayer13_norm1(concated_features_30);  concated_features_30 = None
    l__mod___features_denseblock3_denselayer13_relu1 = self.L__mod___features_denseblock3_denselayer13_relu1(l__mod___features_denseblock3_denselayer13_norm1);  l__mod___features_denseblock3_denselayer13_norm1 = None
    bottleneck_output_60 = self.L__mod___features_denseblock3_denselayer13_conv1(l__mod___features_denseblock3_denselayer13_relu1);  l__mod___features_denseblock3_denselayer13_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer13_norm2 = self.L__mod___features_denseblock3_denselayer13_norm2(bottleneck_output_60);  bottleneck_output_60 = None
    l__mod___features_denseblock3_denselayer13_relu2 = self.L__mod___features_denseblock3_denselayer13_relu2(l__mod___features_denseblock3_denselayer13_norm2);  l__mod___features_denseblock3_denselayer13_norm2 = None
    new_features_60 = self.L__mod___features_denseblock3_denselayer13_conv2(l__mod___features_denseblock3_denselayer13_relu2);  l__mod___features_denseblock3_denselayer13_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_31 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer14_norm1 = self.L__mod___features_denseblock3_denselayer14_norm1(concated_features_31);  concated_features_31 = None
    l__mod___features_denseblock3_denselayer14_relu1 = self.L__mod___features_denseblock3_denselayer14_relu1(l__mod___features_denseblock3_denselayer14_norm1);  l__mod___features_denseblock3_denselayer14_norm1 = None
    bottleneck_output_62 = self.L__mod___features_denseblock3_denselayer14_conv1(l__mod___features_denseblock3_denselayer14_relu1);  l__mod___features_denseblock3_denselayer14_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer14_norm2 = self.L__mod___features_denseblock3_denselayer14_norm2(bottleneck_output_62);  bottleneck_output_62 = None
    l__mod___features_denseblock3_denselayer14_relu2 = self.L__mod___features_denseblock3_denselayer14_relu2(l__mod___features_denseblock3_denselayer14_norm2);  l__mod___features_denseblock3_denselayer14_norm2 = None
    new_features_62 = self.L__mod___features_denseblock3_denselayer14_conv2(l__mod___features_denseblock3_denselayer14_relu2);  l__mod___features_denseblock3_denselayer14_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_32 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer15_norm1 = self.L__mod___features_denseblock3_denselayer15_norm1(concated_features_32);  concated_features_32 = None
    l__mod___features_denseblock3_denselayer15_relu1 = self.L__mod___features_denseblock3_denselayer15_relu1(l__mod___features_denseblock3_denselayer15_norm1);  l__mod___features_denseblock3_denselayer15_norm1 = None
    bottleneck_output_64 = self.L__mod___features_denseblock3_denselayer15_conv1(l__mod___features_denseblock3_denselayer15_relu1);  l__mod___features_denseblock3_denselayer15_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer15_norm2 = self.L__mod___features_denseblock3_denselayer15_norm2(bottleneck_output_64);  bottleneck_output_64 = None
    l__mod___features_denseblock3_denselayer15_relu2 = self.L__mod___features_denseblock3_denselayer15_relu2(l__mod___features_denseblock3_denselayer15_norm2);  l__mod___features_denseblock3_denselayer15_norm2 = None
    new_features_64 = self.L__mod___features_denseblock3_denselayer15_conv2(l__mod___features_denseblock3_denselayer15_relu2);  l__mod___features_denseblock3_denselayer15_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_33 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer16_norm1 = self.L__mod___features_denseblock3_denselayer16_norm1(concated_features_33);  concated_features_33 = None
    l__mod___features_denseblock3_denselayer16_relu1 = self.L__mod___features_denseblock3_denselayer16_relu1(l__mod___features_denseblock3_denselayer16_norm1);  l__mod___features_denseblock3_denselayer16_norm1 = None
    bottleneck_output_66 = self.L__mod___features_denseblock3_denselayer16_conv1(l__mod___features_denseblock3_denselayer16_relu1);  l__mod___features_denseblock3_denselayer16_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer16_norm2 = self.L__mod___features_denseblock3_denselayer16_norm2(bottleneck_output_66);  bottleneck_output_66 = None
    l__mod___features_denseblock3_denselayer16_relu2 = self.L__mod___features_denseblock3_denselayer16_relu2(l__mod___features_denseblock3_denselayer16_norm2);  l__mod___features_denseblock3_denselayer16_norm2 = None
    new_features_66 = self.L__mod___features_denseblock3_denselayer16_conv2(l__mod___features_denseblock3_denselayer16_relu2);  l__mod___features_denseblock3_denselayer16_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_34 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer17_norm1 = self.L__mod___features_denseblock3_denselayer17_norm1(concated_features_34);  concated_features_34 = None
    l__mod___features_denseblock3_denselayer17_relu1 = self.L__mod___features_denseblock3_denselayer17_relu1(l__mod___features_denseblock3_denselayer17_norm1);  l__mod___features_denseblock3_denselayer17_norm1 = None
    bottleneck_output_68 = self.L__mod___features_denseblock3_denselayer17_conv1(l__mod___features_denseblock3_denselayer17_relu1);  l__mod___features_denseblock3_denselayer17_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer17_norm2 = self.L__mod___features_denseblock3_denselayer17_norm2(bottleneck_output_68);  bottleneck_output_68 = None
    l__mod___features_denseblock3_denselayer17_relu2 = self.L__mod___features_denseblock3_denselayer17_relu2(l__mod___features_denseblock3_denselayer17_norm2);  l__mod___features_denseblock3_denselayer17_norm2 = None
    new_features_68 = self.L__mod___features_denseblock3_denselayer17_conv2(l__mod___features_denseblock3_denselayer17_relu2);  l__mod___features_denseblock3_denselayer17_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_35 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer18_norm1 = self.L__mod___features_denseblock3_denselayer18_norm1(concated_features_35);  concated_features_35 = None
    l__mod___features_denseblock3_denselayer18_relu1 = self.L__mod___features_denseblock3_denselayer18_relu1(l__mod___features_denseblock3_denselayer18_norm1);  l__mod___features_denseblock3_denselayer18_norm1 = None
    bottleneck_output_70 = self.L__mod___features_denseblock3_denselayer18_conv1(l__mod___features_denseblock3_denselayer18_relu1);  l__mod___features_denseblock3_denselayer18_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer18_norm2 = self.L__mod___features_denseblock3_denselayer18_norm2(bottleneck_output_70);  bottleneck_output_70 = None
    l__mod___features_denseblock3_denselayer18_relu2 = self.L__mod___features_denseblock3_denselayer18_relu2(l__mod___features_denseblock3_denselayer18_norm2);  l__mod___features_denseblock3_denselayer18_norm2 = None
    new_features_70 = self.L__mod___features_denseblock3_denselayer18_conv2(l__mod___features_denseblock3_denselayer18_relu2);  l__mod___features_denseblock3_denselayer18_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_36 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer19_norm1 = self.L__mod___features_denseblock3_denselayer19_norm1(concated_features_36);  concated_features_36 = None
    l__mod___features_denseblock3_denselayer19_relu1 = self.L__mod___features_denseblock3_denselayer19_relu1(l__mod___features_denseblock3_denselayer19_norm1);  l__mod___features_denseblock3_denselayer19_norm1 = None
    bottleneck_output_72 = self.L__mod___features_denseblock3_denselayer19_conv1(l__mod___features_denseblock3_denselayer19_relu1);  l__mod___features_denseblock3_denselayer19_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer19_norm2 = self.L__mod___features_denseblock3_denselayer19_norm2(bottleneck_output_72);  bottleneck_output_72 = None
    l__mod___features_denseblock3_denselayer19_relu2 = self.L__mod___features_denseblock3_denselayer19_relu2(l__mod___features_denseblock3_denselayer19_norm2);  l__mod___features_denseblock3_denselayer19_norm2 = None
    new_features_72 = self.L__mod___features_denseblock3_denselayer19_conv2(l__mod___features_denseblock3_denselayer19_relu2);  l__mod___features_denseblock3_denselayer19_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_37 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer20_norm1 = self.L__mod___features_denseblock3_denselayer20_norm1(concated_features_37);  concated_features_37 = None
    l__mod___features_denseblock3_denselayer20_relu1 = self.L__mod___features_denseblock3_denselayer20_relu1(l__mod___features_denseblock3_denselayer20_norm1);  l__mod___features_denseblock3_denselayer20_norm1 = None
    bottleneck_output_74 = self.L__mod___features_denseblock3_denselayer20_conv1(l__mod___features_denseblock3_denselayer20_relu1);  l__mod___features_denseblock3_denselayer20_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer20_norm2 = self.L__mod___features_denseblock3_denselayer20_norm2(bottleneck_output_74);  bottleneck_output_74 = None
    l__mod___features_denseblock3_denselayer20_relu2 = self.L__mod___features_denseblock3_denselayer20_relu2(l__mod___features_denseblock3_denselayer20_norm2);  l__mod___features_denseblock3_denselayer20_norm2 = None
    new_features_74 = self.L__mod___features_denseblock3_denselayer20_conv2(l__mod___features_denseblock3_denselayer20_relu2);  l__mod___features_denseblock3_denselayer20_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_38 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72, new_features_74], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer21_norm1 = self.L__mod___features_denseblock3_denselayer21_norm1(concated_features_38);  concated_features_38 = None
    l__mod___features_denseblock3_denselayer21_relu1 = self.L__mod___features_denseblock3_denselayer21_relu1(l__mod___features_denseblock3_denselayer21_norm1);  l__mod___features_denseblock3_denselayer21_norm1 = None
    bottleneck_output_76 = self.L__mod___features_denseblock3_denselayer21_conv1(l__mod___features_denseblock3_denselayer21_relu1);  l__mod___features_denseblock3_denselayer21_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer21_norm2 = self.L__mod___features_denseblock3_denselayer21_norm2(bottleneck_output_76);  bottleneck_output_76 = None
    l__mod___features_denseblock3_denselayer21_relu2 = self.L__mod___features_denseblock3_denselayer21_relu2(l__mod___features_denseblock3_denselayer21_norm2);  l__mod___features_denseblock3_denselayer21_norm2 = None
    new_features_76 = self.L__mod___features_denseblock3_denselayer21_conv2(l__mod___features_denseblock3_denselayer21_relu2);  l__mod___features_denseblock3_denselayer21_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_39 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72, new_features_74, new_features_76], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer22_norm1 = self.L__mod___features_denseblock3_denselayer22_norm1(concated_features_39);  concated_features_39 = None
    l__mod___features_denseblock3_denselayer22_relu1 = self.L__mod___features_denseblock3_denselayer22_relu1(l__mod___features_denseblock3_denselayer22_norm1);  l__mod___features_denseblock3_denselayer22_norm1 = None
    bottleneck_output_78 = self.L__mod___features_denseblock3_denselayer22_conv1(l__mod___features_denseblock3_denselayer22_relu1);  l__mod___features_denseblock3_denselayer22_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer22_norm2 = self.L__mod___features_denseblock3_denselayer22_norm2(bottleneck_output_78);  bottleneck_output_78 = None
    l__mod___features_denseblock3_denselayer22_relu2 = self.L__mod___features_denseblock3_denselayer22_relu2(l__mod___features_denseblock3_denselayer22_norm2);  l__mod___features_denseblock3_denselayer22_norm2 = None
    new_features_78 = self.L__mod___features_denseblock3_denselayer22_conv2(l__mod___features_denseblock3_denselayer22_relu2);  l__mod___features_denseblock3_denselayer22_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_40 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72, new_features_74, new_features_76, new_features_78], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer23_norm1 = self.L__mod___features_denseblock3_denselayer23_norm1(concated_features_40);  concated_features_40 = None
    l__mod___features_denseblock3_denselayer23_relu1 = self.L__mod___features_denseblock3_denselayer23_relu1(l__mod___features_denseblock3_denselayer23_norm1);  l__mod___features_denseblock3_denselayer23_norm1 = None
    bottleneck_output_80 = self.L__mod___features_denseblock3_denselayer23_conv1(l__mod___features_denseblock3_denselayer23_relu1);  l__mod___features_denseblock3_denselayer23_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer23_norm2 = self.L__mod___features_denseblock3_denselayer23_norm2(bottleneck_output_80);  bottleneck_output_80 = None
    l__mod___features_denseblock3_denselayer23_relu2 = self.L__mod___features_denseblock3_denselayer23_relu2(l__mod___features_denseblock3_denselayer23_norm2);  l__mod___features_denseblock3_denselayer23_norm2 = None
    new_features_80 = self.L__mod___features_denseblock3_denselayer23_conv2(l__mod___features_denseblock3_denselayer23_relu2);  l__mod___features_denseblock3_denselayer23_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_41 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72, new_features_74, new_features_76, new_features_78, new_features_80], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock3_denselayer24_norm1 = self.L__mod___features_denseblock3_denselayer24_norm1(concated_features_41);  concated_features_41 = None
    l__mod___features_denseblock3_denselayer24_relu1 = self.L__mod___features_denseblock3_denselayer24_relu1(l__mod___features_denseblock3_denselayer24_norm1);  l__mod___features_denseblock3_denselayer24_norm1 = None
    bottleneck_output_82 = self.L__mod___features_denseblock3_denselayer24_conv1(l__mod___features_denseblock3_denselayer24_relu1);  l__mod___features_denseblock3_denselayer24_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock3_denselayer24_norm2 = self.L__mod___features_denseblock3_denselayer24_norm2(bottleneck_output_82);  bottleneck_output_82 = None
    l__mod___features_denseblock3_denselayer24_relu2 = self.L__mod___features_denseblock3_denselayer24_relu2(l__mod___features_denseblock3_denselayer24_norm2);  l__mod___features_denseblock3_denselayer24_norm2 = None
    new_features_82 = self.L__mod___features_denseblock3_denselayer24_conv2(l__mod___features_denseblock3_denselayer24_relu2);  l__mod___features_denseblock3_denselayer24_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_44 = torch.cat([l__mod___features_transition2_pool, new_features_36, new_features_38, new_features_40, new_features_42, new_features_44, new_features_46, new_features_48, new_features_50, new_features_52, new_features_54, new_features_56, new_features_58, new_features_60, new_features_62, new_features_64, new_features_66, new_features_68, new_features_70, new_features_72, new_features_74, new_features_76, new_features_78, new_features_80, new_features_82], 1);  l__mod___features_transition2_pool = new_features_36 = new_features_38 = new_features_40 = new_features_42 = new_features_44 = new_features_46 = new_features_48 = new_features_50 = new_features_52 = new_features_54 = new_features_56 = new_features_58 = new_features_60 = new_features_62 = new_features_64 = new_features_66 = new_features_68 = new_features_70 = new_features_72 = new_features_74 = new_features_76 = new_features_78 = new_features_80 = new_features_82 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    l__mod___features_transition3_norm = self.L__mod___features_transition3_norm(cat_44);  cat_44 = None
    l__mod___features_transition3_relu = self.L__mod___features_transition3_relu(l__mod___features_transition3_norm);  l__mod___features_transition3_norm = None
    l__mod___features_transition3_conv = self.L__mod___features_transition3_conv(l__mod___features_transition3_relu);  l__mod___features_transition3_relu = None
    l__mod___features_transition3_pool = self.L__mod___features_transition3_pool(l__mod___features_transition3_conv);  l__mod___features_transition3_conv = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_42 = torch.cat([l__mod___features_transition3_pool], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer1_norm1 = self.L__mod___features_denseblock4_denselayer1_norm1(concated_features_42);  concated_features_42 = None
    l__mod___features_denseblock4_denselayer1_relu1 = self.L__mod___features_denseblock4_denselayer1_relu1(l__mod___features_denseblock4_denselayer1_norm1);  l__mod___features_denseblock4_denselayer1_norm1 = None
    bottleneck_output_84 = self.L__mod___features_denseblock4_denselayer1_conv1(l__mod___features_denseblock4_denselayer1_relu1);  l__mod___features_denseblock4_denselayer1_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer1_norm2 = self.L__mod___features_denseblock4_denselayer1_norm2(bottleneck_output_84);  bottleneck_output_84 = None
    l__mod___features_denseblock4_denselayer1_relu2 = self.L__mod___features_denseblock4_denselayer1_relu2(l__mod___features_denseblock4_denselayer1_norm2);  l__mod___features_denseblock4_denselayer1_norm2 = None
    new_features_84 = self.L__mod___features_denseblock4_denselayer1_conv2(l__mod___features_denseblock4_denselayer1_relu2);  l__mod___features_denseblock4_denselayer1_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_43 = torch.cat([l__mod___features_transition3_pool, new_features_84], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer2_norm1 = self.L__mod___features_denseblock4_denselayer2_norm1(concated_features_43);  concated_features_43 = None
    l__mod___features_denseblock4_denselayer2_relu1 = self.L__mod___features_denseblock4_denselayer2_relu1(l__mod___features_denseblock4_denselayer2_norm1);  l__mod___features_denseblock4_denselayer2_norm1 = None
    bottleneck_output_86 = self.L__mod___features_denseblock4_denselayer2_conv1(l__mod___features_denseblock4_denselayer2_relu1);  l__mod___features_denseblock4_denselayer2_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer2_norm2 = self.L__mod___features_denseblock4_denselayer2_norm2(bottleneck_output_86);  bottleneck_output_86 = None
    l__mod___features_denseblock4_denselayer2_relu2 = self.L__mod___features_denseblock4_denselayer2_relu2(l__mod___features_denseblock4_denselayer2_norm2);  l__mod___features_denseblock4_denselayer2_norm2 = None
    new_features_86 = self.L__mod___features_denseblock4_denselayer2_conv2(l__mod___features_denseblock4_denselayer2_relu2);  l__mod___features_denseblock4_denselayer2_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_44 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer3_norm1 = self.L__mod___features_denseblock4_denselayer3_norm1(concated_features_44);  concated_features_44 = None
    l__mod___features_denseblock4_denselayer3_relu1 = self.L__mod___features_denseblock4_denselayer3_relu1(l__mod___features_denseblock4_denselayer3_norm1);  l__mod___features_denseblock4_denselayer3_norm1 = None
    bottleneck_output_88 = self.L__mod___features_denseblock4_denselayer3_conv1(l__mod___features_denseblock4_denselayer3_relu1);  l__mod___features_denseblock4_denselayer3_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer3_norm2 = self.L__mod___features_denseblock4_denselayer3_norm2(bottleneck_output_88);  bottleneck_output_88 = None
    l__mod___features_denseblock4_denselayer3_relu2 = self.L__mod___features_denseblock4_denselayer3_relu2(l__mod___features_denseblock4_denselayer3_norm2);  l__mod___features_denseblock4_denselayer3_norm2 = None
    new_features_88 = self.L__mod___features_denseblock4_denselayer3_conv2(l__mod___features_denseblock4_denselayer3_relu2);  l__mod___features_denseblock4_denselayer3_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_45 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer4_norm1 = self.L__mod___features_denseblock4_denselayer4_norm1(concated_features_45);  concated_features_45 = None
    l__mod___features_denseblock4_denselayer4_relu1 = self.L__mod___features_denseblock4_denselayer4_relu1(l__mod___features_denseblock4_denselayer4_norm1);  l__mod___features_denseblock4_denselayer4_norm1 = None
    bottleneck_output_90 = self.L__mod___features_denseblock4_denselayer4_conv1(l__mod___features_denseblock4_denselayer4_relu1);  l__mod___features_denseblock4_denselayer4_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer4_norm2 = self.L__mod___features_denseblock4_denselayer4_norm2(bottleneck_output_90);  bottleneck_output_90 = None
    l__mod___features_denseblock4_denselayer4_relu2 = self.L__mod___features_denseblock4_denselayer4_relu2(l__mod___features_denseblock4_denselayer4_norm2);  l__mod___features_denseblock4_denselayer4_norm2 = None
    new_features_90 = self.L__mod___features_denseblock4_denselayer4_conv2(l__mod___features_denseblock4_denselayer4_relu2);  l__mod___features_denseblock4_denselayer4_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_46 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer5_norm1 = self.L__mod___features_denseblock4_denselayer5_norm1(concated_features_46);  concated_features_46 = None
    l__mod___features_denseblock4_denselayer5_relu1 = self.L__mod___features_denseblock4_denselayer5_relu1(l__mod___features_denseblock4_denselayer5_norm1);  l__mod___features_denseblock4_denselayer5_norm1 = None
    bottleneck_output_92 = self.L__mod___features_denseblock4_denselayer5_conv1(l__mod___features_denseblock4_denselayer5_relu1);  l__mod___features_denseblock4_denselayer5_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer5_norm2 = self.L__mod___features_denseblock4_denselayer5_norm2(bottleneck_output_92);  bottleneck_output_92 = None
    l__mod___features_denseblock4_denselayer5_relu2 = self.L__mod___features_denseblock4_denselayer5_relu2(l__mod___features_denseblock4_denselayer5_norm2);  l__mod___features_denseblock4_denselayer5_norm2 = None
    new_features_92 = self.L__mod___features_denseblock4_denselayer5_conv2(l__mod___features_denseblock4_denselayer5_relu2);  l__mod___features_denseblock4_denselayer5_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_47 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer6_norm1 = self.L__mod___features_denseblock4_denselayer6_norm1(concated_features_47);  concated_features_47 = None
    l__mod___features_denseblock4_denselayer6_relu1 = self.L__mod___features_denseblock4_denselayer6_relu1(l__mod___features_denseblock4_denselayer6_norm1);  l__mod___features_denseblock4_denselayer6_norm1 = None
    bottleneck_output_94 = self.L__mod___features_denseblock4_denselayer6_conv1(l__mod___features_denseblock4_denselayer6_relu1);  l__mod___features_denseblock4_denselayer6_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer6_norm2 = self.L__mod___features_denseblock4_denselayer6_norm2(bottleneck_output_94);  bottleneck_output_94 = None
    l__mod___features_denseblock4_denselayer6_relu2 = self.L__mod___features_denseblock4_denselayer6_relu2(l__mod___features_denseblock4_denselayer6_norm2);  l__mod___features_denseblock4_denselayer6_norm2 = None
    new_features_94 = self.L__mod___features_denseblock4_denselayer6_conv2(l__mod___features_denseblock4_denselayer6_relu2);  l__mod___features_denseblock4_denselayer6_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_48 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer7_norm1 = self.L__mod___features_denseblock4_denselayer7_norm1(concated_features_48);  concated_features_48 = None
    l__mod___features_denseblock4_denselayer7_relu1 = self.L__mod___features_denseblock4_denselayer7_relu1(l__mod___features_denseblock4_denselayer7_norm1);  l__mod___features_denseblock4_denselayer7_norm1 = None
    bottleneck_output_96 = self.L__mod___features_denseblock4_denselayer7_conv1(l__mod___features_denseblock4_denselayer7_relu1);  l__mod___features_denseblock4_denselayer7_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer7_norm2 = self.L__mod___features_denseblock4_denselayer7_norm2(bottleneck_output_96);  bottleneck_output_96 = None
    l__mod___features_denseblock4_denselayer7_relu2 = self.L__mod___features_denseblock4_denselayer7_relu2(l__mod___features_denseblock4_denselayer7_norm2);  l__mod___features_denseblock4_denselayer7_norm2 = None
    new_features_96 = self.L__mod___features_denseblock4_denselayer7_conv2(l__mod___features_denseblock4_denselayer7_relu2);  l__mod___features_denseblock4_denselayer7_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_49 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer8_norm1 = self.L__mod___features_denseblock4_denselayer8_norm1(concated_features_49);  concated_features_49 = None
    l__mod___features_denseblock4_denselayer8_relu1 = self.L__mod___features_denseblock4_denselayer8_relu1(l__mod___features_denseblock4_denselayer8_norm1);  l__mod___features_denseblock4_denselayer8_norm1 = None
    bottleneck_output_98 = self.L__mod___features_denseblock4_denselayer8_conv1(l__mod___features_denseblock4_denselayer8_relu1);  l__mod___features_denseblock4_denselayer8_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer8_norm2 = self.L__mod___features_denseblock4_denselayer8_norm2(bottleneck_output_98);  bottleneck_output_98 = None
    l__mod___features_denseblock4_denselayer8_relu2 = self.L__mod___features_denseblock4_denselayer8_relu2(l__mod___features_denseblock4_denselayer8_norm2);  l__mod___features_denseblock4_denselayer8_norm2 = None
    new_features_98 = self.L__mod___features_denseblock4_denselayer8_conv2(l__mod___features_denseblock4_denselayer8_relu2);  l__mod___features_denseblock4_denselayer8_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_50 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer9_norm1 = self.L__mod___features_denseblock4_denselayer9_norm1(concated_features_50);  concated_features_50 = None
    l__mod___features_denseblock4_denselayer9_relu1 = self.L__mod___features_denseblock4_denselayer9_relu1(l__mod___features_denseblock4_denselayer9_norm1);  l__mod___features_denseblock4_denselayer9_norm1 = None
    bottleneck_output_100 = self.L__mod___features_denseblock4_denselayer9_conv1(l__mod___features_denseblock4_denselayer9_relu1);  l__mod___features_denseblock4_denselayer9_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer9_norm2 = self.L__mod___features_denseblock4_denselayer9_norm2(bottleneck_output_100);  bottleneck_output_100 = None
    l__mod___features_denseblock4_denselayer9_relu2 = self.L__mod___features_denseblock4_denselayer9_relu2(l__mod___features_denseblock4_denselayer9_norm2);  l__mod___features_denseblock4_denselayer9_norm2 = None
    new_features_100 = self.L__mod___features_denseblock4_denselayer9_conv2(l__mod___features_denseblock4_denselayer9_relu2);  l__mod___features_denseblock4_denselayer9_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_51 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer10_norm1 = self.L__mod___features_denseblock4_denselayer10_norm1(concated_features_51);  concated_features_51 = None
    l__mod___features_denseblock4_denselayer10_relu1 = self.L__mod___features_denseblock4_denselayer10_relu1(l__mod___features_denseblock4_denselayer10_norm1);  l__mod___features_denseblock4_denselayer10_norm1 = None
    bottleneck_output_102 = self.L__mod___features_denseblock4_denselayer10_conv1(l__mod___features_denseblock4_denselayer10_relu1);  l__mod___features_denseblock4_denselayer10_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer10_norm2 = self.L__mod___features_denseblock4_denselayer10_norm2(bottleneck_output_102);  bottleneck_output_102 = None
    l__mod___features_denseblock4_denselayer10_relu2 = self.L__mod___features_denseblock4_denselayer10_relu2(l__mod___features_denseblock4_denselayer10_norm2);  l__mod___features_denseblock4_denselayer10_norm2 = None
    new_features_102 = self.L__mod___features_denseblock4_denselayer10_conv2(l__mod___features_denseblock4_denselayer10_relu2);  l__mod___features_denseblock4_denselayer10_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_52 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer11_norm1 = self.L__mod___features_denseblock4_denselayer11_norm1(concated_features_52);  concated_features_52 = None
    l__mod___features_denseblock4_denselayer11_relu1 = self.L__mod___features_denseblock4_denselayer11_relu1(l__mod___features_denseblock4_denselayer11_norm1);  l__mod___features_denseblock4_denselayer11_norm1 = None
    bottleneck_output_104 = self.L__mod___features_denseblock4_denselayer11_conv1(l__mod___features_denseblock4_denselayer11_relu1);  l__mod___features_denseblock4_denselayer11_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer11_norm2 = self.L__mod___features_denseblock4_denselayer11_norm2(bottleneck_output_104);  bottleneck_output_104 = None
    l__mod___features_denseblock4_denselayer11_relu2 = self.L__mod___features_denseblock4_denselayer11_relu2(l__mod___features_denseblock4_denselayer11_norm2);  l__mod___features_denseblock4_denselayer11_norm2 = None
    new_features_104 = self.L__mod___features_denseblock4_denselayer11_conv2(l__mod___features_denseblock4_denselayer11_relu2);  l__mod___features_denseblock4_denselayer11_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_53 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer12_norm1 = self.L__mod___features_denseblock4_denselayer12_norm1(concated_features_53);  concated_features_53 = None
    l__mod___features_denseblock4_denselayer12_relu1 = self.L__mod___features_denseblock4_denselayer12_relu1(l__mod___features_denseblock4_denselayer12_norm1);  l__mod___features_denseblock4_denselayer12_norm1 = None
    bottleneck_output_106 = self.L__mod___features_denseblock4_denselayer12_conv1(l__mod___features_denseblock4_denselayer12_relu1);  l__mod___features_denseblock4_denselayer12_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer12_norm2 = self.L__mod___features_denseblock4_denselayer12_norm2(bottleneck_output_106);  bottleneck_output_106 = None
    l__mod___features_denseblock4_denselayer12_relu2 = self.L__mod___features_denseblock4_denselayer12_relu2(l__mod___features_denseblock4_denselayer12_norm2);  l__mod___features_denseblock4_denselayer12_norm2 = None
    new_features_106 = self.L__mod___features_denseblock4_denselayer12_conv2(l__mod___features_denseblock4_denselayer12_relu2);  l__mod___features_denseblock4_denselayer12_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_54 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104, new_features_106], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer13_norm1 = self.L__mod___features_denseblock4_denselayer13_norm1(concated_features_54);  concated_features_54 = None
    l__mod___features_denseblock4_denselayer13_relu1 = self.L__mod___features_denseblock4_denselayer13_relu1(l__mod___features_denseblock4_denselayer13_norm1);  l__mod___features_denseblock4_denselayer13_norm1 = None
    bottleneck_output_108 = self.L__mod___features_denseblock4_denselayer13_conv1(l__mod___features_denseblock4_denselayer13_relu1);  l__mod___features_denseblock4_denselayer13_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer13_norm2 = self.L__mod___features_denseblock4_denselayer13_norm2(bottleneck_output_108);  bottleneck_output_108 = None
    l__mod___features_denseblock4_denselayer13_relu2 = self.L__mod___features_denseblock4_denselayer13_relu2(l__mod___features_denseblock4_denselayer13_norm2);  l__mod___features_denseblock4_denselayer13_norm2 = None
    new_features_108 = self.L__mod___features_denseblock4_denselayer13_conv2(l__mod___features_denseblock4_denselayer13_relu2);  l__mod___features_denseblock4_denselayer13_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_55 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104, new_features_106, new_features_108], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer14_norm1 = self.L__mod___features_denseblock4_denselayer14_norm1(concated_features_55);  concated_features_55 = None
    l__mod___features_denseblock4_denselayer14_relu1 = self.L__mod___features_denseblock4_denselayer14_relu1(l__mod___features_denseblock4_denselayer14_norm1);  l__mod___features_denseblock4_denselayer14_norm1 = None
    bottleneck_output_110 = self.L__mod___features_denseblock4_denselayer14_conv1(l__mod___features_denseblock4_denselayer14_relu1);  l__mod___features_denseblock4_denselayer14_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer14_norm2 = self.L__mod___features_denseblock4_denselayer14_norm2(bottleneck_output_110);  bottleneck_output_110 = None
    l__mod___features_denseblock4_denselayer14_relu2 = self.L__mod___features_denseblock4_denselayer14_relu2(l__mod___features_denseblock4_denselayer14_norm2);  l__mod___features_denseblock4_denselayer14_norm2 = None
    new_features_110 = self.L__mod___features_denseblock4_denselayer14_conv2(l__mod___features_denseblock4_denselayer14_relu2);  l__mod___features_denseblock4_denselayer14_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_56 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104, new_features_106, new_features_108, new_features_110], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer15_norm1 = self.L__mod___features_denseblock4_denselayer15_norm1(concated_features_56);  concated_features_56 = None
    l__mod___features_denseblock4_denselayer15_relu1 = self.L__mod___features_denseblock4_denselayer15_relu1(l__mod___features_denseblock4_denselayer15_norm1);  l__mod___features_denseblock4_denselayer15_norm1 = None
    bottleneck_output_112 = self.L__mod___features_denseblock4_denselayer15_conv1(l__mod___features_denseblock4_denselayer15_relu1);  l__mod___features_denseblock4_denselayer15_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer15_norm2 = self.L__mod___features_denseblock4_denselayer15_norm2(bottleneck_output_112);  bottleneck_output_112 = None
    l__mod___features_denseblock4_denselayer15_relu2 = self.L__mod___features_denseblock4_denselayer15_relu2(l__mod___features_denseblock4_denselayer15_norm2);  l__mod___features_denseblock4_denselayer15_norm2 = None
    new_features_112 = self.L__mod___features_denseblock4_denselayer15_conv2(l__mod___features_denseblock4_denselayer15_relu2);  l__mod___features_denseblock4_denselayer15_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    concated_features_57 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104, new_features_106, new_features_108, new_features_110, new_features_112], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    l__mod___features_denseblock4_denselayer16_norm1 = self.L__mod___features_denseblock4_denselayer16_norm1(concated_features_57);  concated_features_57 = None
    l__mod___features_denseblock4_denselayer16_relu1 = self.L__mod___features_denseblock4_denselayer16_relu1(l__mod___features_denseblock4_denselayer16_norm1);  l__mod___features_denseblock4_denselayer16_norm1 = None
    bottleneck_output_114 = self.L__mod___features_denseblock4_denselayer16_conv1(l__mod___features_denseblock4_denselayer16_relu1);  l__mod___features_denseblock4_denselayer16_relu1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    l__mod___features_denseblock4_denselayer16_norm2 = self.L__mod___features_denseblock4_denselayer16_norm2(bottleneck_output_114);  bottleneck_output_114 = None
    l__mod___features_denseblock4_denselayer16_relu2 = self.L__mod___features_denseblock4_denselayer16_relu2(l__mod___features_denseblock4_denselayer16_norm2);  l__mod___features_denseblock4_denselayer16_norm2 = None
    new_features_114 = self.L__mod___features_denseblock4_denselayer16_conv2(l__mod___features_denseblock4_denselayer16_relu2);  l__mod___features_denseblock4_denselayer16_relu2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_61 = torch.cat([l__mod___features_transition3_pool, new_features_84, new_features_86, new_features_88, new_features_90, new_features_92, new_features_94, new_features_96, new_features_98, new_features_100, new_features_102, new_features_104, new_features_106, new_features_108, new_features_110, new_features_112, new_features_114], 1);  l__mod___features_transition3_pool = new_features_84 = new_features_86 = new_features_88 = new_features_90 = new_features_92 = new_features_94 = new_features_96 = new_features_98 = new_features_100 = new_features_102 = new_features_104 = new_features_106 = new_features_108 = new_features_110 = new_features_112 = new_features_114 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    features = self.L__mod___features_norm5(cat_61);  cat_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:214, code: out = F.relu(features, inplace=True)
    out = torch.nn.functional.relu(features, inplace = True);  features = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:215, code: out = F.adaptive_avg_pool2d(out, (1, 1))
    out_1 = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1));  out = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:216, code: out = torch.flatten(out, 1)
    out_2 = torch.flatten(out_1, 1);  out_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:217, code: out = self.classifier(out)
    out_3 = self.L__mod___classifier(out_2);  out_2 = None
    return (out_3,)
    