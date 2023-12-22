from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_start_positions_ : torch.Tensor, L_cloned_inputs_end_positions_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_start_positions_ = L_cloned_inputs_start_positions_
    l_cloned_inputs_end_positions_ = L_cloned_inputs_end_positions_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:582, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_cloned_inputs_input_ids_.view(-1, 128);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:605, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = torch.arange(0, 128, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:606, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze = position_ids.unsqueeze(0);  position_ids = None
    position_ids_1 = unsqueeze.view(-1, 128);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:635, code: inputs_embeds = self.wte(input_ids)
    inputs_embeds = self.L__mod___transformer_wte(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:643, code: hidden_states = self.drop(hidden_states)
    residual = self.L__mod___transformer_drop(inputs_embeds);  inputs_embeds = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states = self.L__mod___transformer_h_0_ln_1(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query = self.L__mod___transformer_h_0_attn_q_proj(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key = self.L__mod___transformer_h_0_attn_k_proj(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value = self.L__mod___transformer_h_0_attn_v_proj(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_1 = query.view((1, 128, 16, 256));  query = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_1 = key.view((1, 128, 16, 256));  key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_2 = value.view((1, 128, 16, 256));  value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_1 = tensor_2.permute(0, 2, 1, 3);  tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions = self.L__mod___transformer_h_0_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_1 = embed_positions.repeat(1, 1, 1);  embed_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_1 = position_ids_1.unsqueeze(-1)
    repeated_position_ids = unsqueeze_1.repeat(1, 1, 64);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos = torch.gather(embed_positions_1, 1, repeated_position_ids);  embed_positions_1 = repeated_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split = torch.functional.split(sincos, 32, dim = -1);  sincos = None
    sin = split[0]
    cos = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot = key_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass = key_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot = query_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass = query_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_6 = sin[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_1 = torch.repeat_interleave(getitem_6, 2, 3);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_7 = cos[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_1 = torch.repeat_interleave(getitem_7, 2, 3);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul = k_rot * cos_1;  cos_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1 = k_rot[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2 = k_rot[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg = -x2;  x2 = None
    x = torch.stack((neg, x1), dim = -1);  neg = x1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten = x.flatten(-2);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_1 = flatten * sin_1;  flatten = sin_1 = None
    k_rot_1 = mul + mul_1;  mul = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_10 = sin[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin = None
    sin_2 = torch.repeat_interleave(getitem_10, 2, 3);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_11 = cos[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos = None
    cos_2 = torch.repeat_interleave(getitem_11, 2, 3);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_2 = q_rot * cos_2;  cos_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_1 = q_rot[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_1 = q_rot[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_1 = -x2_1;  x2_1 = None
    x_1 = torch.stack((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_1 = x_1.flatten(-2);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_3 = flatten_1 * sin_2;  flatten_1 = sin_2 = None
    q_rot_1 = mul_2 + mul_3;  mul_2 = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_2 = torch.cat([k_rot_1, k_pass], dim = -1);  k_rot_1 = k_pass = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_2 = torch.cat([q_rot_1, q_pass], dim = -1);  q_rot_1 = q_pass = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_3 = key_2.permute(0, 2, 1, 3);  key_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_3 = query_2.permute(0, 2, 1, 3);  query_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_0_attn_bias = self.L__mod___transformer_h_0_attn_bias
    causal_mask = l__mod___transformer_h_0_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_0_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_4 = query_3.to(torch.float32);  query_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_4 = key_3.to(torch.float32);  key_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose = key_4.transpose(-1, -2);  key_4 = None
    attn_weights = torch.matmul(query_4, transpose);  query_4 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_3 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value = tensor_3.to(device(type='cpu'));  tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_1 = torch.where(causal_mask, attn_weights, mask_value);  causal_mask = attn_weights = mask_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_0_attn_scale_attn = self.L__mod___transformer_h_0_attn_scale_attn
    attn_weights_2 = attn_weights_1 / l__mod___transformer_h_0_attn_scale_attn;  attn_weights_1 = l__mod___transformer_h_0_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_4 = attn_weights_3.to(torch.float32);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_6 = self.L__mod___transformer_h_0_attn_attn_dropout(attn_weights_4);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output = torch.matmul(attn_weights_6, value_1);  attn_weights_6 = value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_3 = attn_output.permute(0, 2, 1, 3);  attn_output = None
    tensor_4 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_1 = tensor_4.view((1, 128, 4096));  tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_2 = self.L__mod___transformer_h_0_attn_out_proj(attn_output_1);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_4 = self.L__mod___transformer_h_0_attn_resid_dropout(attn_output_2);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_1 = self.L__mod___transformer_h_0_mlp_fc_in(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4 = 0.5 * hidden_states_1
    pow_1 = torch.pow(hidden_states_1, 3.0)
    mul_5 = 0.044715 * pow_1;  pow_1 = None
    add_2 = hidden_states_1 + mul_5;  hidden_states_1 = mul_5 = None
    mul_6 = 0.7978845608028654 * add_2;  add_2 = None
    tanh = torch.tanh(mul_6);  mul_6 = None
    add_3 = 1.0 + tanh;  tanh = None
    hidden_states_2 = mul_4 * add_3;  mul_4 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_3 = self.L__mod___transformer_h_0_mlp_fc_out(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states = self.L__mod___transformer_h_0_mlp_dropout(hidden_states_3);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_4 = attn_output_4 + feed_forward_hidden_states;  attn_output_4 = feed_forward_hidden_states = None
    residual_1 = add_4 + residual;  add_4 = residual = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_6 = self.L__mod___transformer_h_1_ln_1(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_5 = self.L__mod___transformer_h_1_attn_q_proj(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_5 = self.L__mod___transformer_h_1_attn_k_proj(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_2 = self.L__mod___transformer_h_1_attn_v_proj(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_6 = query_5.view((1, 128, 16, 256));  query_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_6 = key_5.view((1, 128, 16, 256));  key_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_7 = value_2.view((1, 128, 16, 256));  value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_3 = tensor_7.permute(0, 2, 1, 3);  tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_2 = self.L__mod___transformer_h_1_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_3 = embed_positions_2.repeat(1, 1, 1);  embed_positions_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_2 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_1 = unsqueeze_2.repeat(1, 1, 64);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_1 = torch.gather(embed_positions_3, 1, repeated_position_ids_1);  embed_positions_3 = repeated_position_ids_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_1 = torch.functional.split(sincos_1, 32, dim = -1);  sincos_1 = None
    sin_3 = split_1[0]
    cos_3 = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_2 = key_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_1 = key_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_2 = query_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_1 = query_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_21 = sin_3[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_4 = torch.repeat_interleave(getitem_21, 2, 3);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_22 = cos_3[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_4 = torch.repeat_interleave(getitem_22, 2, 3);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_8 = k_rot_2 * cos_4;  cos_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_2 = k_rot_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_2 = k_rot_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_2 = -x2_2;  x2_2 = None
    x_2 = torch.stack((neg_2, x1_2), dim = -1);  neg_2 = x1_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_2 = x_2.flatten(-2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_9 = flatten_2 * sin_4;  flatten_2 = sin_4 = None
    k_rot_3 = mul_8 + mul_9;  mul_8 = mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_25 = sin_3[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_3 = None
    sin_5 = torch.repeat_interleave(getitem_25, 2, 3);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_26 = cos_3[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_3 = None
    cos_5 = torch.repeat_interleave(getitem_26, 2, 3);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_10 = q_rot_2 * cos_5;  cos_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_3 = q_rot_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_3 = q_rot_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_3 = -x2_3;  x2_3 = None
    x_3 = torch.stack((neg_3, x1_3), dim = -1);  neg_3 = x1_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_3 = x_3.flatten(-2);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_11 = flatten_3 * sin_5;  flatten_3 = sin_5 = None
    q_rot_3 = mul_10 + mul_11;  mul_10 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_7 = torch.cat([k_rot_3, k_pass_1], dim = -1);  k_rot_3 = k_pass_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_7 = torch.cat([q_rot_3, q_pass_1], dim = -1);  q_rot_3 = q_pass_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_8 = key_7.permute(0, 2, 1, 3);  key_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_8 = query_7.permute(0, 2, 1, 3);  query_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_1_attn_bias = self.L__mod___transformer_h_1_attn_bias
    causal_mask_1 = l__mod___transformer_h_1_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_1_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_9 = query_8.to(torch.float32);  query_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_9 = key_8.to(torch.float32);  key_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_1 = key_9.transpose(-1, -2);  key_9 = None
    attn_weights_7 = torch.matmul(query_9, transpose_1);  query_9 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_8 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_1 = tensor_8.to(device(type='cpu'));  tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_8 = torch.where(causal_mask_1, attn_weights_7, mask_value_1);  causal_mask_1 = attn_weights_7 = mask_value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_1_attn_scale_attn = self.L__mod___transformer_h_1_attn_scale_attn
    attn_weights_9 = attn_weights_8 / l__mod___transformer_h_1_attn_scale_attn;  attn_weights_8 = l__mod___transformer_h_1_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_10 = torch.nn.functional.softmax(attn_weights_9, dim = -1);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_11 = attn_weights_10.to(torch.float32);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_13 = self.L__mod___transformer_h_1_attn_attn_dropout(attn_weights_11);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_6 = torch.matmul(attn_weights_13, value_3);  attn_weights_13 = value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7 = attn_output_6.permute(0, 2, 1, 3);  attn_output_6 = None
    tensor_9 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_7 = tensor_9.view((1, 128, 4096));  tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_8 = self.L__mod___transformer_h_1_attn_out_proj(attn_output_7);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_10 = self.L__mod___transformer_h_1_attn_resid_dropout(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_7 = self.L__mod___transformer_h_1_mlp_fc_in(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12 = 0.5 * hidden_states_7
    pow_2 = torch.pow(hidden_states_7, 3.0)
    mul_13 = 0.044715 * pow_2;  pow_2 = None
    add_8 = hidden_states_7 + mul_13;  hidden_states_7 = mul_13 = None
    mul_14 = 0.7978845608028654 * add_8;  add_8 = None
    tanh_1 = torch.tanh(mul_14);  mul_14 = None
    add_9 = 1.0 + tanh_1;  tanh_1 = None
    hidden_states_8 = mul_12 * add_9;  mul_12 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_9 = self.L__mod___transformer_h_1_mlp_fc_out(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_1 = self.L__mod___transformer_h_1_mlp_dropout(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_10 = attn_output_10 + feed_forward_hidden_states_1;  attn_output_10 = feed_forward_hidden_states_1 = None
    residual_2 = add_10 + residual_1;  add_10 = residual_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_12 = self.L__mod___transformer_h_2_ln_1(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_10 = self.L__mod___transformer_h_2_attn_q_proj(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_10 = self.L__mod___transformer_h_2_attn_k_proj(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_4 = self.L__mod___transformer_h_2_attn_v_proj(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_11 = query_10.view((1, 128, 16, 256));  query_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_11 = key_10.view((1, 128, 16, 256));  key_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_12 = value_4.view((1, 128, 16, 256));  value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_5 = tensor_12.permute(0, 2, 1, 3);  tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_4 = self.L__mod___transformer_h_2_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_5 = embed_positions_4.repeat(1, 1, 1);  embed_positions_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_3 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_2 = unsqueeze_3.repeat(1, 1, 64);  unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_2 = torch.gather(embed_positions_5, 1, repeated_position_ids_2);  embed_positions_5 = repeated_position_ids_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_2 = torch.functional.split(sincos_2, 32, dim = -1);  sincos_2 = None
    sin_6 = split_2[0]
    cos_6 = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_4 = key_11[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_2 = key_11[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_4 = query_11[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_2 = query_11[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_36 = sin_6[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_7 = torch.repeat_interleave(getitem_36, 2, 3);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_37 = cos_6[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_7 = torch.repeat_interleave(getitem_37, 2, 3);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_16 = k_rot_4 * cos_7;  cos_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_4 = k_rot_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_4 = k_rot_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_4 = -x2_4;  x2_4 = None
    x_4 = torch.stack((neg_4, x1_4), dim = -1);  neg_4 = x1_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_4 = x_4.flatten(-2);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_17 = flatten_4 * sin_7;  flatten_4 = sin_7 = None
    k_rot_5 = mul_16 + mul_17;  mul_16 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_40 = sin_6[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_6 = None
    sin_8 = torch.repeat_interleave(getitem_40, 2, 3);  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_41 = cos_6[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_6 = None
    cos_8 = torch.repeat_interleave(getitem_41, 2, 3);  getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_18 = q_rot_4 * cos_8;  cos_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_5 = q_rot_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_5 = q_rot_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_5 = -x2_5;  x2_5 = None
    x_5 = torch.stack((neg_5, x1_5), dim = -1);  neg_5 = x1_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_5 = x_5.flatten(-2);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_19 = flatten_5 * sin_8;  flatten_5 = sin_8 = None
    q_rot_5 = mul_18 + mul_19;  mul_18 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_12 = torch.cat([k_rot_5, k_pass_2], dim = -1);  k_rot_5 = k_pass_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_12 = torch.cat([q_rot_5, q_pass_2], dim = -1);  q_rot_5 = q_pass_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_13 = key_12.permute(0, 2, 1, 3);  key_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_13 = query_12.permute(0, 2, 1, 3);  query_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_2_attn_bias = self.L__mod___transformer_h_2_attn_bias
    causal_mask_2 = l__mod___transformer_h_2_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_2_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_14 = query_13.to(torch.float32);  query_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_14 = key_13.to(torch.float32);  key_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_2 = key_14.transpose(-1, -2);  key_14 = None
    attn_weights_14 = torch.matmul(query_14, transpose_2);  query_14 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_13 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_2 = tensor_13.to(device(type='cpu'));  tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_15 = torch.where(causal_mask_2, attn_weights_14, mask_value_2);  causal_mask_2 = attn_weights_14 = mask_value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_2_attn_scale_attn = self.L__mod___transformer_h_2_attn_scale_attn
    attn_weights_16 = attn_weights_15 / l__mod___transformer_h_2_attn_scale_attn;  attn_weights_15 = l__mod___transformer_h_2_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim = -1);  attn_weights_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_18 = attn_weights_17.to(torch.float32);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_20 = self.L__mod___transformer_h_2_attn_attn_dropout(attn_weights_18);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_12 = torch.matmul(attn_weights_20, value_5);  attn_weights_20 = value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_11 = attn_output_12.permute(0, 2, 1, 3);  attn_output_12 = None
    tensor_14 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_13 = tensor_14.view((1, 128, 4096));  tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_14 = self.L__mod___transformer_h_2_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_16 = self.L__mod___transformer_h_2_attn_resid_dropout(attn_output_14);  attn_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_13 = self.L__mod___transformer_h_2_mlp_fc_in(hidden_states_12);  hidden_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20 = 0.5 * hidden_states_13
    pow_3 = torch.pow(hidden_states_13, 3.0)
    mul_21 = 0.044715 * pow_3;  pow_3 = None
    add_14 = hidden_states_13 + mul_21;  hidden_states_13 = mul_21 = None
    mul_22 = 0.7978845608028654 * add_14;  add_14 = None
    tanh_2 = torch.tanh(mul_22);  mul_22 = None
    add_15 = 1.0 + tanh_2;  tanh_2 = None
    hidden_states_14 = mul_20 * add_15;  mul_20 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_15 = self.L__mod___transformer_h_2_mlp_fc_out(hidden_states_14);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_2 = self.L__mod___transformer_h_2_mlp_dropout(hidden_states_15);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_16 = attn_output_16 + feed_forward_hidden_states_2;  attn_output_16 = feed_forward_hidden_states_2 = None
    residual_3 = add_16 + residual_2;  add_16 = residual_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_18 = self.L__mod___transformer_h_3_ln_1(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_15 = self.L__mod___transformer_h_3_attn_q_proj(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_15 = self.L__mod___transformer_h_3_attn_k_proj(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_6 = self.L__mod___transformer_h_3_attn_v_proj(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_16 = query_15.view((1, 128, 16, 256));  query_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_16 = key_15.view((1, 128, 16, 256));  key_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_17 = value_6.view((1, 128, 16, 256));  value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_7 = tensor_17.permute(0, 2, 1, 3);  tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_6 = self.L__mod___transformer_h_3_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_7 = embed_positions_6.repeat(1, 1, 1);  embed_positions_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_4 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_3 = unsqueeze_4.repeat(1, 1, 64);  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_3 = torch.gather(embed_positions_7, 1, repeated_position_ids_3);  embed_positions_7 = repeated_position_ids_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_3 = torch.functional.split(sincos_3, 32, dim = -1);  sincos_3 = None
    sin_9 = split_3[0]
    cos_9 = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_6 = key_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_3 = key_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_6 = query_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_3 = query_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_51 = sin_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_10 = torch.repeat_interleave(getitem_51, 2, 3);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_52 = cos_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_10 = torch.repeat_interleave(getitem_52, 2, 3);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_24 = k_rot_6 * cos_10;  cos_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_6 = k_rot_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_6 = k_rot_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_6 = -x2_6;  x2_6 = None
    x_6 = torch.stack((neg_6, x1_6), dim = -1);  neg_6 = x1_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_6 = x_6.flatten(-2);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_25 = flatten_6 * sin_10;  flatten_6 = sin_10 = None
    k_rot_7 = mul_24 + mul_25;  mul_24 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_55 = sin_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_9 = None
    sin_11 = torch.repeat_interleave(getitem_55, 2, 3);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_56 = cos_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_9 = None
    cos_11 = torch.repeat_interleave(getitem_56, 2, 3);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_26 = q_rot_6 * cos_11;  cos_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_7 = q_rot_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_7 = q_rot_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_7 = -x2_7;  x2_7 = None
    x_7 = torch.stack((neg_7, x1_7), dim = -1);  neg_7 = x1_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_7 = x_7.flatten(-2);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_27 = flatten_7 * sin_11;  flatten_7 = sin_11 = None
    q_rot_7 = mul_26 + mul_27;  mul_26 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_17 = torch.cat([k_rot_7, k_pass_3], dim = -1);  k_rot_7 = k_pass_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_17 = torch.cat([q_rot_7, q_pass_3], dim = -1);  q_rot_7 = q_pass_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_18 = key_17.permute(0, 2, 1, 3);  key_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_18 = query_17.permute(0, 2, 1, 3);  query_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_3_attn_bias = self.L__mod___transformer_h_3_attn_bias
    causal_mask_3 = l__mod___transformer_h_3_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_3_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_19 = query_18.to(torch.float32);  query_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_19 = key_18.to(torch.float32);  key_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_3 = key_19.transpose(-1, -2);  key_19 = None
    attn_weights_21 = torch.matmul(query_19, transpose_3);  query_19 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_18 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_3 = tensor_18.to(device(type='cpu'));  tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_22 = torch.where(causal_mask_3, attn_weights_21, mask_value_3);  causal_mask_3 = attn_weights_21 = mask_value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_3_attn_scale_attn = self.L__mod___transformer_h_3_attn_scale_attn
    attn_weights_23 = attn_weights_22 / l__mod___transformer_h_3_attn_scale_attn;  attn_weights_22 = l__mod___transformer_h_3_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_24 = torch.nn.functional.softmax(attn_weights_23, dim = -1);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_25 = attn_weights_24.to(torch.float32);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_27 = self.L__mod___transformer_h_3_attn_attn_dropout(attn_weights_25);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_18 = torch.matmul(attn_weights_27, value_7);  attn_weights_27 = value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_15 = attn_output_18.permute(0, 2, 1, 3);  attn_output_18 = None
    tensor_19 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_19 = tensor_19.view((1, 128, 4096));  tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_20 = self.L__mod___transformer_h_3_attn_out_proj(attn_output_19);  attn_output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_22 = self.L__mod___transformer_h_3_attn_resid_dropout(attn_output_20);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_19 = self.L__mod___transformer_h_3_mlp_fc_in(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28 = 0.5 * hidden_states_19
    pow_4 = torch.pow(hidden_states_19, 3.0)
    mul_29 = 0.044715 * pow_4;  pow_4 = None
    add_20 = hidden_states_19 + mul_29;  hidden_states_19 = mul_29 = None
    mul_30 = 0.7978845608028654 * add_20;  add_20 = None
    tanh_3 = torch.tanh(mul_30);  mul_30 = None
    add_21 = 1.0 + tanh_3;  tanh_3 = None
    hidden_states_20 = mul_28 * add_21;  mul_28 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_21 = self.L__mod___transformer_h_3_mlp_fc_out(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_3 = self.L__mod___transformer_h_3_mlp_dropout(hidden_states_21);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_22 = attn_output_22 + feed_forward_hidden_states_3;  attn_output_22 = feed_forward_hidden_states_3 = None
    residual_4 = add_22 + residual_3;  add_22 = residual_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_24 = self.L__mod___transformer_h_4_ln_1(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_20 = self.L__mod___transformer_h_4_attn_q_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_20 = self.L__mod___transformer_h_4_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_8 = self.L__mod___transformer_h_4_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_21 = query_20.view((1, 128, 16, 256));  query_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_21 = key_20.view((1, 128, 16, 256));  key_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_22 = value_8.view((1, 128, 16, 256));  value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_9 = tensor_22.permute(0, 2, 1, 3);  tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_8 = self.L__mod___transformer_h_4_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_9 = embed_positions_8.repeat(1, 1, 1);  embed_positions_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_5 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_4 = unsqueeze_5.repeat(1, 1, 64);  unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_4 = torch.gather(embed_positions_9, 1, repeated_position_ids_4);  embed_positions_9 = repeated_position_ids_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_4 = torch.functional.split(sincos_4, 32, dim = -1);  sincos_4 = None
    sin_12 = split_4[0]
    cos_12 = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_8 = key_21[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_4 = key_21[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_8 = query_21[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_4 = query_21[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_66 = sin_12[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_13 = torch.repeat_interleave(getitem_66, 2, 3);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_67 = cos_12[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_13 = torch.repeat_interleave(getitem_67, 2, 3);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_32 = k_rot_8 * cos_13;  cos_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_8 = k_rot_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_8 = k_rot_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_8 = -x2_8;  x2_8 = None
    x_8 = torch.stack((neg_8, x1_8), dim = -1);  neg_8 = x1_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_8 = x_8.flatten(-2);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_33 = flatten_8 * sin_13;  flatten_8 = sin_13 = None
    k_rot_9 = mul_32 + mul_33;  mul_32 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_70 = sin_12[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_12 = None
    sin_14 = torch.repeat_interleave(getitem_70, 2, 3);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_71 = cos_12[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_12 = None
    cos_14 = torch.repeat_interleave(getitem_71, 2, 3);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_34 = q_rot_8 * cos_14;  cos_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_9 = q_rot_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_9 = q_rot_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_9 = -x2_9;  x2_9 = None
    x_9 = torch.stack((neg_9, x1_9), dim = -1);  neg_9 = x1_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_9 = x_9.flatten(-2);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_35 = flatten_9 * sin_14;  flatten_9 = sin_14 = None
    q_rot_9 = mul_34 + mul_35;  mul_34 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_22 = torch.cat([k_rot_9, k_pass_4], dim = -1);  k_rot_9 = k_pass_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_22 = torch.cat([q_rot_9, q_pass_4], dim = -1);  q_rot_9 = q_pass_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_23 = key_22.permute(0, 2, 1, 3);  key_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_23 = query_22.permute(0, 2, 1, 3);  query_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_4_attn_bias = self.L__mod___transformer_h_4_attn_bias
    causal_mask_4 = l__mod___transformer_h_4_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_4_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_24 = query_23.to(torch.float32);  query_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_24 = key_23.to(torch.float32);  key_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_4 = key_24.transpose(-1, -2);  key_24 = None
    attn_weights_28 = torch.matmul(query_24, transpose_4);  query_24 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_23 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_4 = tensor_23.to(device(type='cpu'));  tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_29 = torch.where(causal_mask_4, attn_weights_28, mask_value_4);  causal_mask_4 = attn_weights_28 = mask_value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_4_attn_scale_attn = self.L__mod___transformer_h_4_attn_scale_attn
    attn_weights_30 = attn_weights_29 / l__mod___transformer_h_4_attn_scale_attn;  attn_weights_29 = l__mod___transformer_h_4_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim = -1);  attn_weights_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_32 = attn_weights_31.to(torch.float32);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_34 = self.L__mod___transformer_h_4_attn_attn_dropout(attn_weights_32);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_24 = torch.matmul(attn_weights_34, value_9);  attn_weights_34 = value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19 = attn_output_24.permute(0, 2, 1, 3);  attn_output_24 = None
    tensor_24 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_25 = tensor_24.view((1, 128, 4096));  tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_26 = self.L__mod___transformer_h_4_attn_out_proj(attn_output_25);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_28 = self.L__mod___transformer_h_4_attn_resid_dropout(attn_output_26);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_25 = self.L__mod___transformer_h_4_mlp_fc_in(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36 = 0.5 * hidden_states_25
    pow_5 = torch.pow(hidden_states_25, 3.0)
    mul_37 = 0.044715 * pow_5;  pow_5 = None
    add_26 = hidden_states_25 + mul_37;  hidden_states_25 = mul_37 = None
    mul_38 = 0.7978845608028654 * add_26;  add_26 = None
    tanh_4 = torch.tanh(mul_38);  mul_38 = None
    add_27 = 1.0 + tanh_4;  tanh_4 = None
    hidden_states_26 = mul_36 * add_27;  mul_36 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_27 = self.L__mod___transformer_h_4_mlp_fc_out(hidden_states_26);  hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_4 = self.L__mod___transformer_h_4_mlp_dropout(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_28 = attn_output_28 + feed_forward_hidden_states_4;  attn_output_28 = feed_forward_hidden_states_4 = None
    residual_5 = add_28 + residual_4;  add_28 = residual_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_30 = self.L__mod___transformer_h_5_ln_1(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_25 = self.L__mod___transformer_h_5_attn_q_proj(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_25 = self.L__mod___transformer_h_5_attn_k_proj(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_10 = self.L__mod___transformer_h_5_attn_v_proj(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_26 = query_25.view((1, 128, 16, 256));  query_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_26 = key_25.view((1, 128, 16, 256));  key_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_27 = value_10.view((1, 128, 16, 256));  value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_11 = tensor_27.permute(0, 2, 1, 3);  tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_10 = self.L__mod___transformer_h_5_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_11 = embed_positions_10.repeat(1, 1, 1);  embed_positions_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_6 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_5 = unsqueeze_6.repeat(1, 1, 64);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_5 = torch.gather(embed_positions_11, 1, repeated_position_ids_5);  embed_positions_11 = repeated_position_ids_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_5 = torch.functional.split(sincos_5, 32, dim = -1);  sincos_5 = None
    sin_15 = split_5[0]
    cos_15 = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_10 = key_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_5 = key_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_10 = query_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_5 = query_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_81 = sin_15[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_16 = torch.repeat_interleave(getitem_81, 2, 3);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_82 = cos_15[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_16 = torch.repeat_interleave(getitem_82, 2, 3);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_40 = k_rot_10 * cos_16;  cos_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_10 = k_rot_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_10 = k_rot_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_10 = -x2_10;  x2_10 = None
    x_10 = torch.stack((neg_10, x1_10), dim = -1);  neg_10 = x1_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_10 = x_10.flatten(-2);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_41 = flatten_10 * sin_16;  flatten_10 = sin_16 = None
    k_rot_11 = mul_40 + mul_41;  mul_40 = mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_85 = sin_15[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_15 = None
    sin_17 = torch.repeat_interleave(getitem_85, 2, 3);  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_86 = cos_15[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_15 = None
    cos_17 = torch.repeat_interleave(getitem_86, 2, 3);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_42 = q_rot_10 * cos_17;  cos_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_11 = q_rot_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_11 = q_rot_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_11 = -x2_11;  x2_11 = None
    x_11 = torch.stack((neg_11, x1_11), dim = -1);  neg_11 = x1_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_11 = x_11.flatten(-2);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_43 = flatten_11 * sin_17;  flatten_11 = sin_17 = None
    q_rot_11 = mul_42 + mul_43;  mul_42 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_27 = torch.cat([k_rot_11, k_pass_5], dim = -1);  k_rot_11 = k_pass_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_27 = torch.cat([q_rot_11, q_pass_5], dim = -1);  q_rot_11 = q_pass_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_28 = key_27.permute(0, 2, 1, 3);  key_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_28 = query_27.permute(0, 2, 1, 3);  query_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_5_attn_bias = self.L__mod___transformer_h_5_attn_bias
    causal_mask_5 = l__mod___transformer_h_5_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_5_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_29 = query_28.to(torch.float32);  query_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_29 = key_28.to(torch.float32);  key_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_5 = key_29.transpose(-1, -2);  key_29 = None
    attn_weights_35 = torch.matmul(query_29, transpose_5);  query_29 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_28 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_5 = tensor_28.to(device(type='cpu'));  tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_36 = torch.where(causal_mask_5, attn_weights_35, mask_value_5);  causal_mask_5 = attn_weights_35 = mask_value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_5_attn_scale_attn = self.L__mod___transformer_h_5_attn_scale_attn
    attn_weights_37 = attn_weights_36 / l__mod___transformer_h_5_attn_scale_attn;  attn_weights_36 = l__mod___transformer_h_5_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_38 = torch.nn.functional.softmax(attn_weights_37, dim = -1);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_39 = attn_weights_38.to(torch.float32);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_41 = self.L__mod___transformer_h_5_attn_attn_dropout(attn_weights_39);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_30 = torch.matmul(attn_weights_41, value_11);  attn_weights_41 = value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_23 = attn_output_30.permute(0, 2, 1, 3);  attn_output_30 = None
    tensor_29 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_31 = tensor_29.view((1, 128, 4096));  tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_32 = self.L__mod___transformer_h_5_attn_out_proj(attn_output_31);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_34 = self.L__mod___transformer_h_5_attn_resid_dropout(attn_output_32);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_31 = self.L__mod___transformer_h_5_mlp_fc_in(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44 = 0.5 * hidden_states_31
    pow_6 = torch.pow(hidden_states_31, 3.0)
    mul_45 = 0.044715 * pow_6;  pow_6 = None
    add_32 = hidden_states_31 + mul_45;  hidden_states_31 = mul_45 = None
    mul_46 = 0.7978845608028654 * add_32;  add_32 = None
    tanh_5 = torch.tanh(mul_46);  mul_46 = None
    add_33 = 1.0 + tanh_5;  tanh_5 = None
    hidden_states_32 = mul_44 * add_33;  mul_44 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_33 = self.L__mod___transformer_h_5_mlp_fc_out(hidden_states_32);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_5 = self.L__mod___transformer_h_5_mlp_dropout(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_34 = attn_output_34 + feed_forward_hidden_states_5;  attn_output_34 = feed_forward_hidden_states_5 = None
    residual_6 = add_34 + residual_5;  add_34 = residual_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_36 = self.L__mod___transformer_h_6_ln_1(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_30 = self.L__mod___transformer_h_6_attn_q_proj(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_30 = self.L__mod___transformer_h_6_attn_k_proj(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_12 = self.L__mod___transformer_h_6_attn_v_proj(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_31 = query_30.view((1, 128, 16, 256));  query_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_31 = key_30.view((1, 128, 16, 256));  key_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_32 = value_12.view((1, 128, 16, 256));  value_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_13 = tensor_32.permute(0, 2, 1, 3);  tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_12 = self.L__mod___transformer_h_6_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_13 = embed_positions_12.repeat(1, 1, 1);  embed_positions_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_7 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_6 = unsqueeze_7.repeat(1, 1, 64);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_6 = torch.gather(embed_positions_13, 1, repeated_position_ids_6);  embed_positions_13 = repeated_position_ids_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_6 = torch.functional.split(sincos_6, 32, dim = -1);  sincos_6 = None
    sin_18 = split_6[0]
    cos_18 = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_12 = key_31[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_6 = key_31[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_12 = query_31[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_6 = query_31[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_96 = sin_18[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_19 = torch.repeat_interleave(getitem_96, 2, 3);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_97 = cos_18[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_19 = torch.repeat_interleave(getitem_97, 2, 3);  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_48 = k_rot_12 * cos_19;  cos_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_12 = k_rot_12[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_12 = k_rot_12[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_12 = -x2_12;  x2_12 = None
    x_12 = torch.stack((neg_12, x1_12), dim = -1);  neg_12 = x1_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_12 = x_12.flatten(-2);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_49 = flatten_12 * sin_19;  flatten_12 = sin_19 = None
    k_rot_13 = mul_48 + mul_49;  mul_48 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_100 = sin_18[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_18 = None
    sin_20 = torch.repeat_interleave(getitem_100, 2, 3);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_101 = cos_18[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_18 = None
    cos_20 = torch.repeat_interleave(getitem_101, 2, 3);  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_50 = q_rot_12 * cos_20;  cos_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_13 = q_rot_12[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_13 = q_rot_12[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_13 = -x2_13;  x2_13 = None
    x_13 = torch.stack((neg_13, x1_13), dim = -1);  neg_13 = x1_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_13 = x_13.flatten(-2);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_51 = flatten_13 * sin_20;  flatten_13 = sin_20 = None
    q_rot_13 = mul_50 + mul_51;  mul_50 = mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_32 = torch.cat([k_rot_13, k_pass_6], dim = -1);  k_rot_13 = k_pass_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_32 = torch.cat([q_rot_13, q_pass_6], dim = -1);  q_rot_13 = q_pass_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_33 = key_32.permute(0, 2, 1, 3);  key_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_33 = query_32.permute(0, 2, 1, 3);  query_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_6_attn_bias = self.L__mod___transformer_h_6_attn_bias
    causal_mask_6 = l__mod___transformer_h_6_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_6_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_34 = query_33.to(torch.float32);  query_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_34 = key_33.to(torch.float32);  key_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_6 = key_34.transpose(-1, -2);  key_34 = None
    attn_weights_42 = torch.matmul(query_34, transpose_6);  query_34 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_33 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_6 = tensor_33.to(device(type='cpu'));  tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_43 = torch.where(causal_mask_6, attn_weights_42, mask_value_6);  causal_mask_6 = attn_weights_42 = mask_value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_6_attn_scale_attn = self.L__mod___transformer_h_6_attn_scale_attn
    attn_weights_44 = attn_weights_43 / l__mod___transformer_h_6_attn_scale_attn;  attn_weights_43 = l__mod___transformer_h_6_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim = -1);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_46 = attn_weights_45.to(torch.float32);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_48 = self.L__mod___transformer_h_6_attn_attn_dropout(attn_weights_46);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_36 = torch.matmul(attn_weights_48, value_13);  attn_weights_48 = value_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_27 = attn_output_36.permute(0, 2, 1, 3);  attn_output_36 = None
    tensor_34 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_37 = tensor_34.view((1, 128, 4096));  tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_38 = self.L__mod___transformer_h_6_attn_out_proj(attn_output_37);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_40 = self.L__mod___transformer_h_6_attn_resid_dropout(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_37 = self.L__mod___transformer_h_6_mlp_fc_in(hidden_states_36);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52 = 0.5 * hidden_states_37
    pow_7 = torch.pow(hidden_states_37, 3.0)
    mul_53 = 0.044715 * pow_7;  pow_7 = None
    add_38 = hidden_states_37 + mul_53;  hidden_states_37 = mul_53 = None
    mul_54 = 0.7978845608028654 * add_38;  add_38 = None
    tanh_6 = torch.tanh(mul_54);  mul_54 = None
    add_39 = 1.0 + tanh_6;  tanh_6 = None
    hidden_states_38 = mul_52 * add_39;  mul_52 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_39 = self.L__mod___transformer_h_6_mlp_fc_out(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_6 = self.L__mod___transformer_h_6_mlp_dropout(hidden_states_39);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_40 = attn_output_40 + feed_forward_hidden_states_6;  attn_output_40 = feed_forward_hidden_states_6 = None
    residual_7 = add_40 + residual_6;  add_40 = residual_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_42 = self.L__mod___transformer_h_7_ln_1(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_35 = self.L__mod___transformer_h_7_attn_q_proj(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_35 = self.L__mod___transformer_h_7_attn_k_proj(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_14 = self.L__mod___transformer_h_7_attn_v_proj(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_36 = query_35.view((1, 128, 16, 256));  query_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_36 = key_35.view((1, 128, 16, 256));  key_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_37 = value_14.view((1, 128, 16, 256));  value_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_15 = tensor_37.permute(0, 2, 1, 3);  tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_14 = self.L__mod___transformer_h_7_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_15 = embed_positions_14.repeat(1, 1, 1);  embed_positions_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_8 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_7 = unsqueeze_8.repeat(1, 1, 64);  unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_7 = torch.gather(embed_positions_15, 1, repeated_position_ids_7);  embed_positions_15 = repeated_position_ids_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_7 = torch.functional.split(sincos_7, 32, dim = -1);  sincos_7 = None
    sin_21 = split_7[0]
    cos_21 = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_14 = key_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_7 = key_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_14 = query_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_7 = query_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_111 = sin_21[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_22 = torch.repeat_interleave(getitem_111, 2, 3);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_112 = cos_21[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_22 = torch.repeat_interleave(getitem_112, 2, 3);  getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_56 = k_rot_14 * cos_22;  cos_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_14 = k_rot_14[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_14 = k_rot_14[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_14 = -x2_14;  x2_14 = None
    x_14 = torch.stack((neg_14, x1_14), dim = -1);  neg_14 = x1_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_14 = x_14.flatten(-2);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_57 = flatten_14 * sin_22;  flatten_14 = sin_22 = None
    k_rot_15 = mul_56 + mul_57;  mul_56 = mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_115 = sin_21[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_21 = None
    sin_23 = torch.repeat_interleave(getitem_115, 2, 3);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_116 = cos_21[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_21 = None
    cos_23 = torch.repeat_interleave(getitem_116, 2, 3);  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_58 = q_rot_14 * cos_23;  cos_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_15 = q_rot_14[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_15 = q_rot_14[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_15 = -x2_15;  x2_15 = None
    x_15 = torch.stack((neg_15, x1_15), dim = -1);  neg_15 = x1_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_15 = x_15.flatten(-2);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_59 = flatten_15 * sin_23;  flatten_15 = sin_23 = None
    q_rot_15 = mul_58 + mul_59;  mul_58 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_37 = torch.cat([k_rot_15, k_pass_7], dim = -1);  k_rot_15 = k_pass_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_37 = torch.cat([q_rot_15, q_pass_7], dim = -1);  q_rot_15 = q_pass_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_38 = key_37.permute(0, 2, 1, 3);  key_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_38 = query_37.permute(0, 2, 1, 3);  query_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_7_attn_bias = self.L__mod___transformer_h_7_attn_bias
    causal_mask_7 = l__mod___transformer_h_7_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_7_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_39 = query_38.to(torch.float32);  query_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_39 = key_38.to(torch.float32);  key_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_7 = key_39.transpose(-1, -2);  key_39 = None
    attn_weights_49 = torch.matmul(query_39, transpose_7);  query_39 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_38 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_7 = tensor_38.to(device(type='cpu'));  tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_50 = torch.where(causal_mask_7, attn_weights_49, mask_value_7);  causal_mask_7 = attn_weights_49 = mask_value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_7_attn_scale_attn = self.L__mod___transformer_h_7_attn_scale_attn
    attn_weights_51 = attn_weights_50 / l__mod___transformer_h_7_attn_scale_attn;  attn_weights_50 = l__mod___transformer_h_7_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_52 = torch.nn.functional.softmax(attn_weights_51, dim = -1);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_53 = attn_weights_52.to(torch.float32);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_55 = self.L__mod___transformer_h_7_attn_attn_dropout(attn_weights_53);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_42 = torch.matmul(attn_weights_55, value_15);  attn_weights_55 = value_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_31 = attn_output_42.permute(0, 2, 1, 3);  attn_output_42 = None
    tensor_39 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_43 = tensor_39.view((1, 128, 4096));  tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_44 = self.L__mod___transformer_h_7_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_46 = self.L__mod___transformer_h_7_attn_resid_dropout(attn_output_44);  attn_output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_43 = self.L__mod___transformer_h_7_mlp_fc_in(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60 = 0.5 * hidden_states_43
    pow_8 = torch.pow(hidden_states_43, 3.0)
    mul_61 = 0.044715 * pow_8;  pow_8 = None
    add_44 = hidden_states_43 + mul_61;  hidden_states_43 = mul_61 = None
    mul_62 = 0.7978845608028654 * add_44;  add_44 = None
    tanh_7 = torch.tanh(mul_62);  mul_62 = None
    add_45 = 1.0 + tanh_7;  tanh_7 = None
    hidden_states_44 = mul_60 * add_45;  mul_60 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_45 = self.L__mod___transformer_h_7_mlp_fc_out(hidden_states_44);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_7 = self.L__mod___transformer_h_7_mlp_dropout(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_46 = attn_output_46 + feed_forward_hidden_states_7;  attn_output_46 = feed_forward_hidden_states_7 = None
    residual_8 = add_46 + residual_7;  add_46 = residual_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_48 = self.L__mod___transformer_h_8_ln_1(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_40 = self.L__mod___transformer_h_8_attn_q_proj(hidden_states_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_40 = self.L__mod___transformer_h_8_attn_k_proj(hidden_states_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_16 = self.L__mod___transformer_h_8_attn_v_proj(hidden_states_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_41 = query_40.view((1, 128, 16, 256));  query_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_41 = key_40.view((1, 128, 16, 256));  key_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_42 = value_16.view((1, 128, 16, 256));  value_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_17 = tensor_42.permute(0, 2, 1, 3);  tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_16 = self.L__mod___transformer_h_8_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_17 = embed_positions_16.repeat(1, 1, 1);  embed_positions_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_9 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_8 = unsqueeze_9.repeat(1, 1, 64);  unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_8 = torch.gather(embed_positions_17, 1, repeated_position_ids_8);  embed_positions_17 = repeated_position_ids_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_8 = torch.functional.split(sincos_8, 32, dim = -1);  sincos_8 = None
    sin_24 = split_8[0]
    cos_24 = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_16 = key_41[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_8 = key_41[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_16 = query_41[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_8 = query_41[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_126 = sin_24[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_25 = torch.repeat_interleave(getitem_126, 2, 3);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_127 = cos_24[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_25 = torch.repeat_interleave(getitem_127, 2, 3);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_64 = k_rot_16 * cos_25;  cos_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_16 = k_rot_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_16 = k_rot_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_16 = -x2_16;  x2_16 = None
    x_16 = torch.stack((neg_16, x1_16), dim = -1);  neg_16 = x1_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_16 = x_16.flatten(-2);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_65 = flatten_16 * sin_25;  flatten_16 = sin_25 = None
    k_rot_17 = mul_64 + mul_65;  mul_64 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_130 = sin_24[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_24 = None
    sin_26 = torch.repeat_interleave(getitem_130, 2, 3);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_131 = cos_24[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_24 = None
    cos_26 = torch.repeat_interleave(getitem_131, 2, 3);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_66 = q_rot_16 * cos_26;  cos_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_17 = q_rot_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_17 = q_rot_16[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_17 = -x2_17;  x2_17 = None
    x_17 = torch.stack((neg_17, x1_17), dim = -1);  neg_17 = x1_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_17 = x_17.flatten(-2);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_67 = flatten_17 * sin_26;  flatten_17 = sin_26 = None
    q_rot_17 = mul_66 + mul_67;  mul_66 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_42 = torch.cat([k_rot_17, k_pass_8], dim = -1);  k_rot_17 = k_pass_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_42 = torch.cat([q_rot_17, q_pass_8], dim = -1);  q_rot_17 = q_pass_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_43 = key_42.permute(0, 2, 1, 3);  key_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_43 = query_42.permute(0, 2, 1, 3);  query_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_8_attn_bias = self.L__mod___transformer_h_8_attn_bias
    causal_mask_8 = l__mod___transformer_h_8_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_8_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_44 = query_43.to(torch.float32);  query_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_44 = key_43.to(torch.float32);  key_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_8 = key_44.transpose(-1, -2);  key_44 = None
    attn_weights_56 = torch.matmul(query_44, transpose_8);  query_44 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_43 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_8 = tensor_43.to(device(type='cpu'));  tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_57 = torch.where(causal_mask_8, attn_weights_56, mask_value_8);  causal_mask_8 = attn_weights_56 = mask_value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_8_attn_scale_attn = self.L__mod___transformer_h_8_attn_scale_attn
    attn_weights_58 = attn_weights_57 / l__mod___transformer_h_8_attn_scale_attn;  attn_weights_57 = l__mod___transformer_h_8_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_60 = attn_weights_59.to(torch.float32);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_62 = self.L__mod___transformer_h_8_attn_attn_dropout(attn_weights_60);  attn_weights_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_48 = torch.matmul(attn_weights_62, value_17);  attn_weights_62 = value_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_35 = attn_output_48.permute(0, 2, 1, 3);  attn_output_48 = None
    tensor_44 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_49 = tensor_44.view((1, 128, 4096));  tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_50 = self.L__mod___transformer_h_8_attn_out_proj(attn_output_49);  attn_output_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_52 = self.L__mod___transformer_h_8_attn_resid_dropout(attn_output_50);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_49 = self.L__mod___transformer_h_8_mlp_fc_in(hidden_states_48);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68 = 0.5 * hidden_states_49
    pow_9 = torch.pow(hidden_states_49, 3.0)
    mul_69 = 0.044715 * pow_9;  pow_9 = None
    add_50 = hidden_states_49 + mul_69;  hidden_states_49 = mul_69 = None
    mul_70 = 0.7978845608028654 * add_50;  add_50 = None
    tanh_8 = torch.tanh(mul_70);  mul_70 = None
    add_51 = 1.0 + tanh_8;  tanh_8 = None
    hidden_states_50 = mul_68 * add_51;  mul_68 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_51 = self.L__mod___transformer_h_8_mlp_fc_out(hidden_states_50);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_8 = self.L__mod___transformer_h_8_mlp_dropout(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_52 = attn_output_52 + feed_forward_hidden_states_8;  attn_output_52 = feed_forward_hidden_states_8 = None
    residual_9 = add_52 + residual_8;  add_52 = residual_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_54 = self.L__mod___transformer_h_9_ln_1(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_45 = self.L__mod___transformer_h_9_attn_q_proj(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_45 = self.L__mod___transformer_h_9_attn_k_proj(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_18 = self.L__mod___transformer_h_9_attn_v_proj(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_46 = query_45.view((1, 128, 16, 256));  query_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_46 = key_45.view((1, 128, 16, 256));  key_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_47 = value_18.view((1, 128, 16, 256));  value_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_19 = tensor_47.permute(0, 2, 1, 3);  tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_18 = self.L__mod___transformer_h_9_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_19 = embed_positions_18.repeat(1, 1, 1);  embed_positions_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_10 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_9 = unsqueeze_10.repeat(1, 1, 64);  unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_9 = torch.gather(embed_positions_19, 1, repeated_position_ids_9);  embed_positions_19 = repeated_position_ids_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_9 = torch.functional.split(sincos_9, 32, dim = -1);  sincos_9 = None
    sin_27 = split_9[0]
    cos_27 = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_18 = key_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_9 = key_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_18 = query_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_9 = query_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_141 = sin_27[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_28 = torch.repeat_interleave(getitem_141, 2, 3);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_142 = cos_27[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_28 = torch.repeat_interleave(getitem_142, 2, 3);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_72 = k_rot_18 * cos_28;  cos_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_18 = k_rot_18[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_18 = k_rot_18[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_18 = -x2_18;  x2_18 = None
    x_18 = torch.stack((neg_18, x1_18), dim = -1);  neg_18 = x1_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_18 = x_18.flatten(-2);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_73 = flatten_18 * sin_28;  flatten_18 = sin_28 = None
    k_rot_19 = mul_72 + mul_73;  mul_72 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_145 = sin_27[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_27 = None
    sin_29 = torch.repeat_interleave(getitem_145, 2, 3);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_146 = cos_27[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_27 = None
    cos_29 = torch.repeat_interleave(getitem_146, 2, 3);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_74 = q_rot_18 * cos_29;  cos_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_19 = q_rot_18[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_19 = q_rot_18[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_19 = -x2_19;  x2_19 = None
    x_19 = torch.stack((neg_19, x1_19), dim = -1);  neg_19 = x1_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_19 = x_19.flatten(-2);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_75 = flatten_19 * sin_29;  flatten_19 = sin_29 = None
    q_rot_19 = mul_74 + mul_75;  mul_74 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_47 = torch.cat([k_rot_19, k_pass_9], dim = -1);  k_rot_19 = k_pass_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_47 = torch.cat([q_rot_19, q_pass_9], dim = -1);  q_rot_19 = q_pass_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_48 = key_47.permute(0, 2, 1, 3);  key_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_48 = query_47.permute(0, 2, 1, 3);  query_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_9_attn_bias = self.L__mod___transformer_h_9_attn_bias
    causal_mask_9 = l__mod___transformer_h_9_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_9_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_49 = query_48.to(torch.float32);  query_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_49 = key_48.to(torch.float32);  key_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_9 = key_49.transpose(-1, -2);  key_49 = None
    attn_weights_63 = torch.matmul(query_49, transpose_9);  query_49 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_48 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_9 = tensor_48.to(device(type='cpu'));  tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_64 = torch.where(causal_mask_9, attn_weights_63, mask_value_9);  causal_mask_9 = attn_weights_63 = mask_value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_9_attn_scale_attn = self.L__mod___transformer_h_9_attn_scale_attn
    attn_weights_65 = attn_weights_64 / l__mod___transformer_h_9_attn_scale_attn;  attn_weights_64 = l__mod___transformer_h_9_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_66 = torch.nn.functional.softmax(attn_weights_65, dim = -1);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_67 = attn_weights_66.to(torch.float32);  attn_weights_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_69 = self.L__mod___transformer_h_9_attn_attn_dropout(attn_weights_67);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_54 = torch.matmul(attn_weights_69, value_19);  attn_weights_69 = value_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39 = attn_output_54.permute(0, 2, 1, 3);  attn_output_54 = None
    tensor_49 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_55 = tensor_49.view((1, 128, 4096));  tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_56 = self.L__mod___transformer_h_9_attn_out_proj(attn_output_55);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_58 = self.L__mod___transformer_h_9_attn_resid_dropout(attn_output_56);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_55 = self.L__mod___transformer_h_9_mlp_fc_in(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76 = 0.5 * hidden_states_55
    pow_10 = torch.pow(hidden_states_55, 3.0)
    mul_77 = 0.044715 * pow_10;  pow_10 = None
    add_56 = hidden_states_55 + mul_77;  hidden_states_55 = mul_77 = None
    mul_78 = 0.7978845608028654 * add_56;  add_56 = None
    tanh_9 = torch.tanh(mul_78);  mul_78 = None
    add_57 = 1.0 + tanh_9;  tanh_9 = None
    hidden_states_56 = mul_76 * add_57;  mul_76 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_57 = self.L__mod___transformer_h_9_mlp_fc_out(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_9 = self.L__mod___transformer_h_9_mlp_dropout(hidden_states_57);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_58 = attn_output_58 + feed_forward_hidden_states_9;  attn_output_58 = feed_forward_hidden_states_9 = None
    residual_10 = add_58 + residual_9;  add_58 = residual_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_60 = self.L__mod___transformer_h_10_ln_1(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_50 = self.L__mod___transformer_h_10_attn_q_proj(hidden_states_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_50 = self.L__mod___transformer_h_10_attn_k_proj(hidden_states_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_20 = self.L__mod___transformer_h_10_attn_v_proj(hidden_states_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_51 = query_50.view((1, 128, 16, 256));  query_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_51 = key_50.view((1, 128, 16, 256));  key_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_52 = value_20.view((1, 128, 16, 256));  value_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_21 = tensor_52.permute(0, 2, 1, 3);  tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_20 = self.L__mod___transformer_h_10_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_21 = embed_positions_20.repeat(1, 1, 1);  embed_positions_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_11 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_10 = unsqueeze_11.repeat(1, 1, 64);  unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_10 = torch.gather(embed_positions_21, 1, repeated_position_ids_10);  embed_positions_21 = repeated_position_ids_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_10 = torch.functional.split(sincos_10, 32, dim = -1);  sincos_10 = None
    sin_30 = split_10[0]
    cos_30 = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_20 = key_51[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_10 = key_51[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_20 = query_51[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_10 = query_51[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_156 = sin_30[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_31 = torch.repeat_interleave(getitem_156, 2, 3);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_157 = cos_30[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_31 = torch.repeat_interleave(getitem_157, 2, 3);  getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_80 = k_rot_20 * cos_31;  cos_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_20 = k_rot_20[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_20 = k_rot_20[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_20 = -x2_20;  x2_20 = None
    x_20 = torch.stack((neg_20, x1_20), dim = -1);  neg_20 = x1_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_20 = x_20.flatten(-2);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_81 = flatten_20 * sin_31;  flatten_20 = sin_31 = None
    k_rot_21 = mul_80 + mul_81;  mul_80 = mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_160 = sin_30[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_30 = None
    sin_32 = torch.repeat_interleave(getitem_160, 2, 3);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_161 = cos_30[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_30 = None
    cos_32 = torch.repeat_interleave(getitem_161, 2, 3);  getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_82 = q_rot_20 * cos_32;  cos_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_21 = q_rot_20[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_21 = q_rot_20[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_21 = -x2_21;  x2_21 = None
    x_21 = torch.stack((neg_21, x1_21), dim = -1);  neg_21 = x1_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_21 = x_21.flatten(-2);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_83 = flatten_21 * sin_32;  flatten_21 = sin_32 = None
    q_rot_21 = mul_82 + mul_83;  mul_82 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_52 = torch.cat([k_rot_21, k_pass_10], dim = -1);  k_rot_21 = k_pass_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_52 = torch.cat([q_rot_21, q_pass_10], dim = -1);  q_rot_21 = q_pass_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_53 = key_52.permute(0, 2, 1, 3);  key_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_53 = query_52.permute(0, 2, 1, 3);  query_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_10_attn_bias = self.L__mod___transformer_h_10_attn_bias
    causal_mask_10 = l__mod___transformer_h_10_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_10_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_54 = query_53.to(torch.float32);  query_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_54 = key_53.to(torch.float32);  key_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_10 = key_54.transpose(-1, -2);  key_54 = None
    attn_weights_70 = torch.matmul(query_54, transpose_10);  query_54 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_53 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_10 = tensor_53.to(device(type='cpu'));  tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_71 = torch.where(causal_mask_10, attn_weights_70, mask_value_10);  causal_mask_10 = attn_weights_70 = mask_value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_10_attn_scale_attn = self.L__mod___transformer_h_10_attn_scale_attn
    attn_weights_72 = attn_weights_71 / l__mod___transformer_h_10_attn_scale_attn;  attn_weights_71 = l__mod___transformer_h_10_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_73 = torch.nn.functional.softmax(attn_weights_72, dim = -1);  attn_weights_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_74 = attn_weights_73.to(torch.float32);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_76 = self.L__mod___transformer_h_10_attn_attn_dropout(attn_weights_74);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_60 = torch.matmul(attn_weights_76, value_21);  attn_weights_76 = value_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_43 = attn_output_60.permute(0, 2, 1, 3);  attn_output_60 = None
    tensor_54 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_61 = tensor_54.view((1, 128, 4096));  tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_62 = self.L__mod___transformer_h_10_attn_out_proj(attn_output_61);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_64 = self.L__mod___transformer_h_10_attn_resid_dropout(attn_output_62);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_61 = self.L__mod___transformer_h_10_mlp_fc_in(hidden_states_60);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84 = 0.5 * hidden_states_61
    pow_11 = torch.pow(hidden_states_61, 3.0)
    mul_85 = 0.044715 * pow_11;  pow_11 = None
    add_62 = hidden_states_61 + mul_85;  hidden_states_61 = mul_85 = None
    mul_86 = 0.7978845608028654 * add_62;  add_62 = None
    tanh_10 = torch.tanh(mul_86);  mul_86 = None
    add_63 = 1.0 + tanh_10;  tanh_10 = None
    hidden_states_62 = mul_84 * add_63;  mul_84 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_63 = self.L__mod___transformer_h_10_mlp_fc_out(hidden_states_62);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_10 = self.L__mod___transformer_h_10_mlp_dropout(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_64 = attn_output_64 + feed_forward_hidden_states_10;  attn_output_64 = feed_forward_hidden_states_10 = None
    residual_11 = add_64 + residual_10;  add_64 = residual_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_66 = self.L__mod___transformer_h_11_ln_1(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_55 = self.L__mod___transformer_h_11_attn_q_proj(hidden_states_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_55 = self.L__mod___transformer_h_11_attn_k_proj(hidden_states_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_22 = self.L__mod___transformer_h_11_attn_v_proj(hidden_states_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_56 = query_55.view((1, 128, 16, 256));  query_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_56 = key_55.view((1, 128, 16, 256));  key_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_57 = value_22.view((1, 128, 16, 256));  value_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_23 = tensor_57.permute(0, 2, 1, 3);  tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_22 = self.L__mod___transformer_h_11_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_23 = embed_positions_22.repeat(1, 1, 1);  embed_positions_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_12 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_11 = unsqueeze_12.repeat(1, 1, 64);  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_11 = torch.gather(embed_positions_23, 1, repeated_position_ids_11);  embed_positions_23 = repeated_position_ids_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_11 = torch.functional.split(sincos_11, 32, dim = -1);  sincos_11 = None
    sin_33 = split_11[0]
    cos_33 = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_22 = key_56[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_11 = key_56[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_22 = query_56[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_11 = query_56[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_171 = sin_33[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_34 = torch.repeat_interleave(getitem_171, 2, 3);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_172 = cos_33[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_34 = torch.repeat_interleave(getitem_172, 2, 3);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_88 = k_rot_22 * cos_34;  cos_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_22 = k_rot_22[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_22 = k_rot_22[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_22 = -x2_22;  x2_22 = None
    x_22 = torch.stack((neg_22, x1_22), dim = -1);  neg_22 = x1_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_22 = x_22.flatten(-2);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_89 = flatten_22 * sin_34;  flatten_22 = sin_34 = None
    k_rot_23 = mul_88 + mul_89;  mul_88 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_175 = sin_33[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_33 = None
    sin_35 = torch.repeat_interleave(getitem_175, 2, 3);  getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_176 = cos_33[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_33 = None
    cos_35 = torch.repeat_interleave(getitem_176, 2, 3);  getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_90 = q_rot_22 * cos_35;  cos_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_23 = q_rot_22[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_23 = q_rot_22[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_23 = -x2_23;  x2_23 = None
    x_23 = torch.stack((neg_23, x1_23), dim = -1);  neg_23 = x1_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_23 = x_23.flatten(-2);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_91 = flatten_23 * sin_35;  flatten_23 = sin_35 = None
    q_rot_23 = mul_90 + mul_91;  mul_90 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_57 = torch.cat([k_rot_23, k_pass_11], dim = -1);  k_rot_23 = k_pass_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_57 = torch.cat([q_rot_23, q_pass_11], dim = -1);  q_rot_23 = q_pass_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_58 = key_57.permute(0, 2, 1, 3);  key_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_58 = query_57.permute(0, 2, 1, 3);  query_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_11_attn_bias = self.L__mod___transformer_h_11_attn_bias
    causal_mask_11 = l__mod___transformer_h_11_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_11_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_59 = query_58.to(torch.float32);  query_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_59 = key_58.to(torch.float32);  key_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_11 = key_59.transpose(-1, -2);  key_59 = None
    attn_weights_77 = torch.matmul(query_59, transpose_11);  query_59 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_58 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_11 = tensor_58.to(device(type='cpu'));  tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_78 = torch.where(causal_mask_11, attn_weights_77, mask_value_11);  causal_mask_11 = attn_weights_77 = mask_value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_11_attn_scale_attn = self.L__mod___transformer_h_11_attn_scale_attn
    attn_weights_79 = attn_weights_78 / l__mod___transformer_h_11_attn_scale_attn;  attn_weights_78 = l__mod___transformer_h_11_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_80 = torch.nn.functional.softmax(attn_weights_79, dim = -1);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_81 = attn_weights_80.to(torch.float32);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_83 = self.L__mod___transformer_h_11_attn_attn_dropout(attn_weights_81);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_66 = torch.matmul(attn_weights_83, value_23);  attn_weights_83 = value_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_47 = attn_output_66.permute(0, 2, 1, 3);  attn_output_66 = None
    tensor_59 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_67 = tensor_59.view((1, 128, 4096));  tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_68 = self.L__mod___transformer_h_11_attn_out_proj(attn_output_67);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_70 = self.L__mod___transformer_h_11_attn_resid_dropout(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_67 = self.L__mod___transformer_h_11_mlp_fc_in(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92 = 0.5 * hidden_states_67
    pow_12 = torch.pow(hidden_states_67, 3.0)
    mul_93 = 0.044715 * pow_12;  pow_12 = None
    add_68 = hidden_states_67 + mul_93;  hidden_states_67 = mul_93 = None
    mul_94 = 0.7978845608028654 * add_68;  add_68 = None
    tanh_11 = torch.tanh(mul_94);  mul_94 = None
    add_69 = 1.0 + tanh_11;  tanh_11 = None
    hidden_states_68 = mul_92 * add_69;  mul_92 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_69 = self.L__mod___transformer_h_11_mlp_fc_out(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_11 = self.L__mod___transformer_h_11_mlp_dropout(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_70 = attn_output_70 + feed_forward_hidden_states_11;  attn_output_70 = feed_forward_hidden_states_11 = None
    residual_12 = add_70 + residual_11;  add_70 = residual_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_72 = self.L__mod___transformer_h_12_ln_1(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_60 = self.L__mod___transformer_h_12_attn_q_proj(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_60 = self.L__mod___transformer_h_12_attn_k_proj(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_24 = self.L__mod___transformer_h_12_attn_v_proj(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_61 = query_60.view((1, 128, 16, 256));  query_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_61 = key_60.view((1, 128, 16, 256));  key_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_62 = value_24.view((1, 128, 16, 256));  value_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_25 = tensor_62.permute(0, 2, 1, 3);  tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_24 = self.L__mod___transformer_h_12_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_25 = embed_positions_24.repeat(1, 1, 1);  embed_positions_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_13 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_12 = unsqueeze_13.repeat(1, 1, 64);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_12 = torch.gather(embed_positions_25, 1, repeated_position_ids_12);  embed_positions_25 = repeated_position_ids_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_12 = torch.functional.split(sincos_12, 32, dim = -1);  sincos_12 = None
    sin_36 = split_12[0]
    cos_36 = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_24 = key_61[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_12 = key_61[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_24 = query_61[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_12 = query_61[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_186 = sin_36[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_37 = torch.repeat_interleave(getitem_186, 2, 3);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_187 = cos_36[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_37 = torch.repeat_interleave(getitem_187, 2, 3);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_96 = k_rot_24 * cos_37;  cos_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_24 = k_rot_24[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_24 = k_rot_24[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_24 = -x2_24;  x2_24 = None
    x_24 = torch.stack((neg_24, x1_24), dim = -1);  neg_24 = x1_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_24 = x_24.flatten(-2);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_97 = flatten_24 * sin_37;  flatten_24 = sin_37 = None
    k_rot_25 = mul_96 + mul_97;  mul_96 = mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_190 = sin_36[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_36 = None
    sin_38 = torch.repeat_interleave(getitem_190, 2, 3);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_191 = cos_36[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_36 = None
    cos_38 = torch.repeat_interleave(getitem_191, 2, 3);  getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_98 = q_rot_24 * cos_38;  cos_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_25 = q_rot_24[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_25 = q_rot_24[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_25 = -x2_25;  x2_25 = None
    x_25 = torch.stack((neg_25, x1_25), dim = -1);  neg_25 = x1_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_25 = x_25.flatten(-2);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_99 = flatten_25 * sin_38;  flatten_25 = sin_38 = None
    q_rot_25 = mul_98 + mul_99;  mul_98 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_62 = torch.cat([k_rot_25, k_pass_12], dim = -1);  k_rot_25 = k_pass_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_62 = torch.cat([q_rot_25, q_pass_12], dim = -1);  q_rot_25 = q_pass_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_63 = key_62.permute(0, 2, 1, 3);  key_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_63 = query_62.permute(0, 2, 1, 3);  query_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_12_attn_bias = self.L__mod___transformer_h_12_attn_bias
    causal_mask_12 = l__mod___transformer_h_12_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_12_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_64 = query_63.to(torch.float32);  query_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_64 = key_63.to(torch.float32);  key_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_12 = key_64.transpose(-1, -2);  key_64 = None
    attn_weights_84 = torch.matmul(query_64, transpose_12);  query_64 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_63 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_12 = tensor_63.to(device(type='cpu'));  tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_85 = torch.where(causal_mask_12, attn_weights_84, mask_value_12);  causal_mask_12 = attn_weights_84 = mask_value_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_12_attn_scale_attn = self.L__mod___transformer_h_12_attn_scale_attn
    attn_weights_86 = attn_weights_85 / l__mod___transformer_h_12_attn_scale_attn;  attn_weights_85 = l__mod___transformer_h_12_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_87 = torch.nn.functional.softmax(attn_weights_86, dim = -1);  attn_weights_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_88 = attn_weights_87.to(torch.float32);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_90 = self.L__mod___transformer_h_12_attn_attn_dropout(attn_weights_88);  attn_weights_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_72 = torch.matmul(attn_weights_90, value_25);  attn_weights_90 = value_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51 = attn_output_72.permute(0, 2, 1, 3);  attn_output_72 = None
    tensor_64 = permute_51.contiguous();  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_73 = tensor_64.view((1, 128, 4096));  tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_74 = self.L__mod___transformer_h_12_attn_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_76 = self.L__mod___transformer_h_12_attn_resid_dropout(attn_output_74);  attn_output_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_73 = self.L__mod___transformer_h_12_mlp_fc_in(hidden_states_72);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100 = 0.5 * hidden_states_73
    pow_13 = torch.pow(hidden_states_73, 3.0)
    mul_101 = 0.044715 * pow_13;  pow_13 = None
    add_74 = hidden_states_73 + mul_101;  hidden_states_73 = mul_101 = None
    mul_102 = 0.7978845608028654 * add_74;  add_74 = None
    tanh_12 = torch.tanh(mul_102);  mul_102 = None
    add_75 = 1.0 + tanh_12;  tanh_12 = None
    hidden_states_74 = mul_100 * add_75;  mul_100 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_75 = self.L__mod___transformer_h_12_mlp_fc_out(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_12 = self.L__mod___transformer_h_12_mlp_dropout(hidden_states_75);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_76 = attn_output_76 + feed_forward_hidden_states_12;  attn_output_76 = feed_forward_hidden_states_12 = None
    residual_13 = add_76 + residual_12;  add_76 = residual_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_78 = self.L__mod___transformer_h_13_ln_1(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_65 = self.L__mod___transformer_h_13_attn_q_proj(hidden_states_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_65 = self.L__mod___transformer_h_13_attn_k_proj(hidden_states_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_26 = self.L__mod___transformer_h_13_attn_v_proj(hidden_states_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_66 = query_65.view((1, 128, 16, 256));  query_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_66 = key_65.view((1, 128, 16, 256));  key_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_67 = value_26.view((1, 128, 16, 256));  value_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_27 = tensor_67.permute(0, 2, 1, 3);  tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_26 = self.L__mod___transformer_h_13_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_27 = embed_positions_26.repeat(1, 1, 1);  embed_positions_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_14 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_13 = unsqueeze_14.repeat(1, 1, 64);  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_13 = torch.gather(embed_positions_27, 1, repeated_position_ids_13);  embed_positions_27 = repeated_position_ids_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_13 = torch.functional.split(sincos_13, 32, dim = -1);  sincos_13 = None
    sin_39 = split_13[0]
    cos_39 = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_26 = key_66[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_13 = key_66[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_26 = query_66[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_13 = query_66[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_201 = sin_39[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_40 = torch.repeat_interleave(getitem_201, 2, 3);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_202 = cos_39[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_40 = torch.repeat_interleave(getitem_202, 2, 3);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_104 = k_rot_26 * cos_40;  cos_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_26 = k_rot_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_26 = k_rot_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_26 = -x2_26;  x2_26 = None
    x_26 = torch.stack((neg_26, x1_26), dim = -1);  neg_26 = x1_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_26 = x_26.flatten(-2);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_105 = flatten_26 * sin_40;  flatten_26 = sin_40 = None
    k_rot_27 = mul_104 + mul_105;  mul_104 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_205 = sin_39[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_39 = None
    sin_41 = torch.repeat_interleave(getitem_205, 2, 3);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_206 = cos_39[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_39 = None
    cos_41 = torch.repeat_interleave(getitem_206, 2, 3);  getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_106 = q_rot_26 * cos_41;  cos_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_27 = q_rot_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_27 = q_rot_26[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_27 = -x2_27;  x2_27 = None
    x_27 = torch.stack((neg_27, x1_27), dim = -1);  neg_27 = x1_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_27 = x_27.flatten(-2);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_107 = flatten_27 * sin_41;  flatten_27 = sin_41 = None
    q_rot_27 = mul_106 + mul_107;  mul_106 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_67 = torch.cat([k_rot_27, k_pass_13], dim = -1);  k_rot_27 = k_pass_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_67 = torch.cat([q_rot_27, q_pass_13], dim = -1);  q_rot_27 = q_pass_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_68 = key_67.permute(0, 2, 1, 3);  key_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_68 = query_67.permute(0, 2, 1, 3);  query_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_13_attn_bias = self.L__mod___transformer_h_13_attn_bias
    causal_mask_13 = l__mod___transformer_h_13_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_13_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_69 = query_68.to(torch.float32);  query_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_69 = key_68.to(torch.float32);  key_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_13 = key_69.transpose(-1, -2);  key_69 = None
    attn_weights_91 = torch.matmul(query_69, transpose_13);  query_69 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_68 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_13 = tensor_68.to(device(type='cpu'));  tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_92 = torch.where(causal_mask_13, attn_weights_91, mask_value_13);  causal_mask_13 = attn_weights_91 = mask_value_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_13_attn_scale_attn = self.L__mod___transformer_h_13_attn_scale_attn
    attn_weights_93 = attn_weights_92 / l__mod___transformer_h_13_attn_scale_attn;  attn_weights_92 = l__mod___transformer_h_13_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_94 = torch.nn.functional.softmax(attn_weights_93, dim = -1);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_95 = attn_weights_94.to(torch.float32);  attn_weights_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_97 = self.L__mod___transformer_h_13_attn_attn_dropout(attn_weights_95);  attn_weights_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_78 = torch.matmul(attn_weights_97, value_27);  attn_weights_97 = value_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_55 = attn_output_78.permute(0, 2, 1, 3);  attn_output_78 = None
    tensor_69 = permute_55.contiguous();  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_79 = tensor_69.view((1, 128, 4096));  tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_80 = self.L__mod___transformer_h_13_attn_out_proj(attn_output_79);  attn_output_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_82 = self.L__mod___transformer_h_13_attn_resid_dropout(attn_output_80);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_79 = self.L__mod___transformer_h_13_mlp_fc_in(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_108 = 0.5 * hidden_states_79
    pow_14 = torch.pow(hidden_states_79, 3.0)
    mul_109 = 0.044715 * pow_14;  pow_14 = None
    add_80 = hidden_states_79 + mul_109;  hidden_states_79 = mul_109 = None
    mul_110 = 0.7978845608028654 * add_80;  add_80 = None
    tanh_13 = torch.tanh(mul_110);  mul_110 = None
    add_81 = 1.0 + tanh_13;  tanh_13 = None
    hidden_states_80 = mul_108 * add_81;  mul_108 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_81 = self.L__mod___transformer_h_13_mlp_fc_out(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_13 = self.L__mod___transformer_h_13_mlp_dropout(hidden_states_81);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_82 = attn_output_82 + feed_forward_hidden_states_13;  attn_output_82 = feed_forward_hidden_states_13 = None
    residual_14 = add_82 + residual_13;  add_82 = residual_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_84 = self.L__mod___transformer_h_14_ln_1(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_70 = self.L__mod___transformer_h_14_attn_q_proj(hidden_states_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_70 = self.L__mod___transformer_h_14_attn_k_proj(hidden_states_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_28 = self.L__mod___transformer_h_14_attn_v_proj(hidden_states_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_71 = query_70.view((1, 128, 16, 256));  query_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_71 = key_70.view((1, 128, 16, 256));  key_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_72 = value_28.view((1, 128, 16, 256));  value_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_29 = tensor_72.permute(0, 2, 1, 3);  tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_28 = self.L__mod___transformer_h_14_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_29 = embed_positions_28.repeat(1, 1, 1);  embed_positions_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_15 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_14 = unsqueeze_15.repeat(1, 1, 64);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_14 = torch.gather(embed_positions_29, 1, repeated_position_ids_14);  embed_positions_29 = repeated_position_ids_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_14 = torch.functional.split(sincos_14, 32, dim = -1);  sincos_14 = None
    sin_42 = split_14[0]
    cos_42 = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_28 = key_71[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_14 = key_71[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_28 = query_71[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_14 = query_71[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_216 = sin_42[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_43 = torch.repeat_interleave(getitem_216, 2, 3);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_217 = cos_42[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_43 = torch.repeat_interleave(getitem_217, 2, 3);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_112 = k_rot_28 * cos_43;  cos_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_28 = k_rot_28[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_28 = k_rot_28[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_28 = -x2_28;  x2_28 = None
    x_28 = torch.stack((neg_28, x1_28), dim = -1);  neg_28 = x1_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_28 = x_28.flatten(-2);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_113 = flatten_28 * sin_43;  flatten_28 = sin_43 = None
    k_rot_29 = mul_112 + mul_113;  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_220 = sin_42[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_42 = None
    sin_44 = torch.repeat_interleave(getitem_220, 2, 3);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_221 = cos_42[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_42 = None
    cos_44 = torch.repeat_interleave(getitem_221, 2, 3);  getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_114 = q_rot_28 * cos_44;  cos_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_29 = q_rot_28[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_29 = q_rot_28[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_29 = -x2_29;  x2_29 = None
    x_29 = torch.stack((neg_29, x1_29), dim = -1);  neg_29 = x1_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_29 = x_29.flatten(-2);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_115 = flatten_29 * sin_44;  flatten_29 = sin_44 = None
    q_rot_29 = mul_114 + mul_115;  mul_114 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_72 = torch.cat([k_rot_29, k_pass_14], dim = -1);  k_rot_29 = k_pass_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_72 = torch.cat([q_rot_29, q_pass_14], dim = -1);  q_rot_29 = q_pass_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_73 = key_72.permute(0, 2, 1, 3);  key_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_73 = query_72.permute(0, 2, 1, 3);  query_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_14_attn_bias = self.L__mod___transformer_h_14_attn_bias
    causal_mask_14 = l__mod___transformer_h_14_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_14_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_74 = query_73.to(torch.float32);  query_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_74 = key_73.to(torch.float32);  key_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_14 = key_74.transpose(-1, -2);  key_74 = None
    attn_weights_98 = torch.matmul(query_74, transpose_14);  query_74 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_73 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_14 = tensor_73.to(device(type='cpu'));  tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_99 = torch.where(causal_mask_14, attn_weights_98, mask_value_14);  causal_mask_14 = attn_weights_98 = mask_value_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_14_attn_scale_attn = self.L__mod___transformer_h_14_attn_scale_attn
    attn_weights_100 = attn_weights_99 / l__mod___transformer_h_14_attn_scale_attn;  attn_weights_99 = l__mod___transformer_h_14_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_101 = torch.nn.functional.softmax(attn_weights_100, dim = -1);  attn_weights_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_102 = attn_weights_101.to(torch.float32);  attn_weights_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_104 = self.L__mod___transformer_h_14_attn_attn_dropout(attn_weights_102);  attn_weights_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_84 = torch.matmul(attn_weights_104, value_29);  attn_weights_104 = value_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59 = attn_output_84.permute(0, 2, 1, 3);  attn_output_84 = None
    tensor_74 = permute_59.contiguous();  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_85 = tensor_74.view((1, 128, 4096));  tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_86 = self.L__mod___transformer_h_14_attn_out_proj(attn_output_85);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_88 = self.L__mod___transformer_h_14_attn_resid_dropout(attn_output_86);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_85 = self.L__mod___transformer_h_14_mlp_fc_in(hidden_states_84);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116 = 0.5 * hidden_states_85
    pow_15 = torch.pow(hidden_states_85, 3.0)
    mul_117 = 0.044715 * pow_15;  pow_15 = None
    add_86 = hidden_states_85 + mul_117;  hidden_states_85 = mul_117 = None
    mul_118 = 0.7978845608028654 * add_86;  add_86 = None
    tanh_14 = torch.tanh(mul_118);  mul_118 = None
    add_87 = 1.0 + tanh_14;  tanh_14 = None
    hidden_states_86 = mul_116 * add_87;  mul_116 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_87 = self.L__mod___transformer_h_14_mlp_fc_out(hidden_states_86);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_14 = self.L__mod___transformer_h_14_mlp_dropout(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_88 = attn_output_88 + feed_forward_hidden_states_14;  attn_output_88 = feed_forward_hidden_states_14 = None
    residual_15 = add_88 + residual_14;  add_88 = residual_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_90 = self.L__mod___transformer_h_15_ln_1(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_75 = self.L__mod___transformer_h_15_attn_q_proj(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_75 = self.L__mod___transformer_h_15_attn_k_proj(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_30 = self.L__mod___transformer_h_15_attn_v_proj(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_76 = query_75.view((1, 128, 16, 256));  query_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_76 = key_75.view((1, 128, 16, 256));  key_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_77 = value_30.view((1, 128, 16, 256));  value_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_31 = tensor_77.permute(0, 2, 1, 3);  tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_30 = self.L__mod___transformer_h_15_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_31 = embed_positions_30.repeat(1, 1, 1);  embed_positions_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_16 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_15 = unsqueeze_16.repeat(1, 1, 64);  unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_15 = torch.gather(embed_positions_31, 1, repeated_position_ids_15);  embed_positions_31 = repeated_position_ids_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_15 = torch.functional.split(sincos_15, 32, dim = -1);  sincos_15 = None
    sin_45 = split_15[0]
    cos_45 = split_15[1];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_30 = key_76[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_15 = key_76[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_30 = query_76[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_15 = query_76[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_231 = sin_45[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_46 = torch.repeat_interleave(getitem_231, 2, 3);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_232 = cos_45[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_46 = torch.repeat_interleave(getitem_232, 2, 3);  getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_120 = k_rot_30 * cos_46;  cos_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_30 = k_rot_30[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_30 = k_rot_30[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_30 = -x2_30;  x2_30 = None
    x_30 = torch.stack((neg_30, x1_30), dim = -1);  neg_30 = x1_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_30 = x_30.flatten(-2);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_121 = flatten_30 * sin_46;  flatten_30 = sin_46 = None
    k_rot_31 = mul_120 + mul_121;  mul_120 = mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_235 = sin_45[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_45 = None
    sin_47 = torch.repeat_interleave(getitem_235, 2, 3);  getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_236 = cos_45[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_45 = None
    cos_47 = torch.repeat_interleave(getitem_236, 2, 3);  getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_122 = q_rot_30 * cos_47;  cos_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_31 = q_rot_30[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_31 = q_rot_30[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_31 = -x2_31;  x2_31 = None
    x_31 = torch.stack((neg_31, x1_31), dim = -1);  neg_31 = x1_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_31 = x_31.flatten(-2);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_123 = flatten_31 * sin_47;  flatten_31 = sin_47 = None
    q_rot_31 = mul_122 + mul_123;  mul_122 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_77 = torch.cat([k_rot_31, k_pass_15], dim = -1);  k_rot_31 = k_pass_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_77 = torch.cat([q_rot_31, q_pass_15], dim = -1);  q_rot_31 = q_pass_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_78 = key_77.permute(0, 2, 1, 3);  key_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_78 = query_77.permute(0, 2, 1, 3);  query_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_15_attn_bias = self.L__mod___transformer_h_15_attn_bias
    causal_mask_15 = l__mod___transformer_h_15_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_15_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_79 = query_78.to(torch.float32);  query_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_79 = key_78.to(torch.float32);  key_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_15 = key_79.transpose(-1, -2);  key_79 = None
    attn_weights_105 = torch.matmul(query_79, transpose_15);  query_79 = transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_78 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_15 = tensor_78.to(device(type='cpu'));  tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_106 = torch.where(causal_mask_15, attn_weights_105, mask_value_15);  causal_mask_15 = attn_weights_105 = mask_value_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_15_attn_scale_attn = self.L__mod___transformer_h_15_attn_scale_attn
    attn_weights_107 = attn_weights_106 / l__mod___transformer_h_15_attn_scale_attn;  attn_weights_106 = l__mod___transformer_h_15_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_108 = torch.nn.functional.softmax(attn_weights_107, dim = -1);  attn_weights_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_109 = attn_weights_108.to(torch.float32);  attn_weights_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_111 = self.L__mod___transformer_h_15_attn_attn_dropout(attn_weights_109);  attn_weights_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_90 = torch.matmul(attn_weights_111, value_31);  attn_weights_111 = value_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_63 = attn_output_90.permute(0, 2, 1, 3);  attn_output_90 = None
    tensor_79 = permute_63.contiguous();  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_91 = tensor_79.view((1, 128, 4096));  tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_92 = self.L__mod___transformer_h_15_attn_out_proj(attn_output_91);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_94 = self.L__mod___transformer_h_15_attn_resid_dropout(attn_output_92);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_91 = self.L__mod___transformer_h_15_mlp_fc_in(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_124 = 0.5 * hidden_states_91
    pow_16 = torch.pow(hidden_states_91, 3.0)
    mul_125 = 0.044715 * pow_16;  pow_16 = None
    add_92 = hidden_states_91 + mul_125;  hidden_states_91 = mul_125 = None
    mul_126 = 0.7978845608028654 * add_92;  add_92 = None
    tanh_15 = torch.tanh(mul_126);  mul_126 = None
    add_93 = 1.0 + tanh_15;  tanh_15 = None
    hidden_states_92 = mul_124 * add_93;  mul_124 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_93 = self.L__mod___transformer_h_15_mlp_fc_out(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_15 = self.L__mod___transformer_h_15_mlp_dropout(hidden_states_93);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_94 = attn_output_94 + feed_forward_hidden_states_15;  attn_output_94 = feed_forward_hidden_states_15 = None
    residual_16 = add_94 + residual_15;  add_94 = residual_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_96 = self.L__mod___transformer_h_16_ln_1(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_80 = self.L__mod___transformer_h_16_attn_q_proj(hidden_states_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_80 = self.L__mod___transformer_h_16_attn_k_proj(hidden_states_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_32 = self.L__mod___transformer_h_16_attn_v_proj(hidden_states_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_81 = query_80.view((1, 128, 16, 256));  query_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_81 = key_80.view((1, 128, 16, 256));  key_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_82 = value_32.view((1, 128, 16, 256));  value_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_33 = tensor_82.permute(0, 2, 1, 3);  tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_32 = self.L__mod___transformer_h_16_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_33 = embed_positions_32.repeat(1, 1, 1);  embed_positions_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_17 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_16 = unsqueeze_17.repeat(1, 1, 64);  unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_16 = torch.gather(embed_positions_33, 1, repeated_position_ids_16);  embed_positions_33 = repeated_position_ids_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_16 = torch.functional.split(sincos_16, 32, dim = -1);  sincos_16 = None
    sin_48 = split_16[0]
    cos_48 = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_32 = key_81[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_16 = key_81[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_32 = query_81[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_16 = query_81[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_246 = sin_48[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_49 = torch.repeat_interleave(getitem_246, 2, 3);  getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_247 = cos_48[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_49 = torch.repeat_interleave(getitem_247, 2, 3);  getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_128 = k_rot_32 * cos_49;  cos_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_32 = k_rot_32[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_32 = k_rot_32[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_32 = -x2_32;  x2_32 = None
    x_32 = torch.stack((neg_32, x1_32), dim = -1);  neg_32 = x1_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_32 = x_32.flatten(-2);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_129 = flatten_32 * sin_49;  flatten_32 = sin_49 = None
    k_rot_33 = mul_128 + mul_129;  mul_128 = mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_250 = sin_48[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_48 = None
    sin_50 = torch.repeat_interleave(getitem_250, 2, 3);  getitem_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_251 = cos_48[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_48 = None
    cos_50 = torch.repeat_interleave(getitem_251, 2, 3);  getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_130 = q_rot_32 * cos_50;  cos_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_33 = q_rot_32[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_33 = q_rot_32[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_33 = -x2_33;  x2_33 = None
    x_33 = torch.stack((neg_33, x1_33), dim = -1);  neg_33 = x1_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_33 = x_33.flatten(-2);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_131 = flatten_33 * sin_50;  flatten_33 = sin_50 = None
    q_rot_33 = mul_130 + mul_131;  mul_130 = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_82 = torch.cat([k_rot_33, k_pass_16], dim = -1);  k_rot_33 = k_pass_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_82 = torch.cat([q_rot_33, q_pass_16], dim = -1);  q_rot_33 = q_pass_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_83 = key_82.permute(0, 2, 1, 3);  key_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_83 = query_82.permute(0, 2, 1, 3);  query_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_16_attn_bias = self.L__mod___transformer_h_16_attn_bias
    causal_mask_16 = l__mod___transformer_h_16_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_16_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_84 = query_83.to(torch.float32);  query_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_84 = key_83.to(torch.float32);  key_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_16 = key_84.transpose(-1, -2);  key_84 = None
    attn_weights_112 = torch.matmul(query_84, transpose_16);  query_84 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_83 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_16 = tensor_83.to(device(type='cpu'));  tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_113 = torch.where(causal_mask_16, attn_weights_112, mask_value_16);  causal_mask_16 = attn_weights_112 = mask_value_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_16_attn_scale_attn = self.L__mod___transformer_h_16_attn_scale_attn
    attn_weights_114 = attn_weights_113 / l__mod___transformer_h_16_attn_scale_attn;  attn_weights_113 = l__mod___transformer_h_16_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_115 = torch.nn.functional.softmax(attn_weights_114, dim = -1);  attn_weights_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_116 = attn_weights_115.to(torch.float32);  attn_weights_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_118 = self.L__mod___transformer_h_16_attn_attn_dropout(attn_weights_116);  attn_weights_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_96 = torch.matmul(attn_weights_118, value_33);  attn_weights_118 = value_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_67 = attn_output_96.permute(0, 2, 1, 3);  attn_output_96 = None
    tensor_84 = permute_67.contiguous();  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_97 = tensor_84.view((1, 128, 4096));  tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_98 = self.L__mod___transformer_h_16_attn_out_proj(attn_output_97);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_100 = self.L__mod___transformer_h_16_attn_resid_dropout(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_97 = self.L__mod___transformer_h_16_mlp_fc_in(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_132 = 0.5 * hidden_states_97
    pow_17 = torch.pow(hidden_states_97, 3.0)
    mul_133 = 0.044715 * pow_17;  pow_17 = None
    add_98 = hidden_states_97 + mul_133;  hidden_states_97 = mul_133 = None
    mul_134 = 0.7978845608028654 * add_98;  add_98 = None
    tanh_16 = torch.tanh(mul_134);  mul_134 = None
    add_99 = 1.0 + tanh_16;  tanh_16 = None
    hidden_states_98 = mul_132 * add_99;  mul_132 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_99 = self.L__mod___transformer_h_16_mlp_fc_out(hidden_states_98);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_16 = self.L__mod___transformer_h_16_mlp_dropout(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_100 = attn_output_100 + feed_forward_hidden_states_16;  attn_output_100 = feed_forward_hidden_states_16 = None
    residual_17 = add_100 + residual_16;  add_100 = residual_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_102 = self.L__mod___transformer_h_17_ln_1(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_85 = self.L__mod___transformer_h_17_attn_q_proj(hidden_states_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_85 = self.L__mod___transformer_h_17_attn_k_proj(hidden_states_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_34 = self.L__mod___transformer_h_17_attn_v_proj(hidden_states_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_86 = query_85.view((1, 128, 16, 256));  query_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_86 = key_85.view((1, 128, 16, 256));  key_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_87 = value_34.view((1, 128, 16, 256));  value_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_35 = tensor_87.permute(0, 2, 1, 3);  tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_34 = self.L__mod___transformer_h_17_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_35 = embed_positions_34.repeat(1, 1, 1);  embed_positions_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_18 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_17 = unsqueeze_18.repeat(1, 1, 64);  unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_17 = torch.gather(embed_positions_35, 1, repeated_position_ids_17);  embed_positions_35 = repeated_position_ids_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_17 = torch.functional.split(sincos_17, 32, dim = -1);  sincos_17 = None
    sin_51 = split_17[0]
    cos_51 = split_17[1];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_34 = key_86[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_17 = key_86[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_34 = query_86[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_17 = query_86[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_261 = sin_51[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_52 = torch.repeat_interleave(getitem_261, 2, 3);  getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_262 = cos_51[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_52 = torch.repeat_interleave(getitem_262, 2, 3);  getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_136 = k_rot_34 * cos_52;  cos_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_34 = k_rot_34[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_34 = k_rot_34[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_34 = -x2_34;  x2_34 = None
    x_34 = torch.stack((neg_34, x1_34), dim = -1);  neg_34 = x1_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_34 = x_34.flatten(-2);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_137 = flatten_34 * sin_52;  flatten_34 = sin_52 = None
    k_rot_35 = mul_136 + mul_137;  mul_136 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_265 = sin_51[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_51 = None
    sin_53 = torch.repeat_interleave(getitem_265, 2, 3);  getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_266 = cos_51[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_51 = None
    cos_53 = torch.repeat_interleave(getitem_266, 2, 3);  getitem_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_138 = q_rot_34 * cos_53;  cos_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_35 = q_rot_34[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_35 = q_rot_34[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_35 = -x2_35;  x2_35 = None
    x_35 = torch.stack((neg_35, x1_35), dim = -1);  neg_35 = x1_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_35 = x_35.flatten(-2);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_139 = flatten_35 * sin_53;  flatten_35 = sin_53 = None
    q_rot_35 = mul_138 + mul_139;  mul_138 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_87 = torch.cat([k_rot_35, k_pass_17], dim = -1);  k_rot_35 = k_pass_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_87 = torch.cat([q_rot_35, q_pass_17], dim = -1);  q_rot_35 = q_pass_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_88 = key_87.permute(0, 2, 1, 3);  key_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_88 = query_87.permute(0, 2, 1, 3);  query_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_17_attn_bias = self.L__mod___transformer_h_17_attn_bias
    causal_mask_17 = l__mod___transformer_h_17_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_17_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_89 = query_88.to(torch.float32);  query_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_89 = key_88.to(torch.float32);  key_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_17 = key_89.transpose(-1, -2);  key_89 = None
    attn_weights_119 = torch.matmul(query_89, transpose_17);  query_89 = transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_88 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_17 = tensor_88.to(device(type='cpu'));  tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_120 = torch.where(causal_mask_17, attn_weights_119, mask_value_17);  causal_mask_17 = attn_weights_119 = mask_value_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_17_attn_scale_attn = self.L__mod___transformer_h_17_attn_scale_attn
    attn_weights_121 = attn_weights_120 / l__mod___transformer_h_17_attn_scale_attn;  attn_weights_120 = l__mod___transformer_h_17_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_122 = torch.nn.functional.softmax(attn_weights_121, dim = -1);  attn_weights_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_123 = attn_weights_122.to(torch.float32);  attn_weights_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_125 = self.L__mod___transformer_h_17_attn_attn_dropout(attn_weights_123);  attn_weights_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_102 = torch.matmul(attn_weights_125, value_35);  attn_weights_125 = value_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_71 = attn_output_102.permute(0, 2, 1, 3);  attn_output_102 = None
    tensor_89 = permute_71.contiguous();  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_103 = tensor_89.view((1, 128, 4096));  tensor_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_104 = self.L__mod___transformer_h_17_attn_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_106 = self.L__mod___transformer_h_17_attn_resid_dropout(attn_output_104);  attn_output_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_103 = self.L__mod___transformer_h_17_mlp_fc_in(hidden_states_102);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_140 = 0.5 * hidden_states_103
    pow_18 = torch.pow(hidden_states_103, 3.0)
    mul_141 = 0.044715 * pow_18;  pow_18 = None
    add_104 = hidden_states_103 + mul_141;  hidden_states_103 = mul_141 = None
    mul_142 = 0.7978845608028654 * add_104;  add_104 = None
    tanh_17 = torch.tanh(mul_142);  mul_142 = None
    add_105 = 1.0 + tanh_17;  tanh_17 = None
    hidden_states_104 = mul_140 * add_105;  mul_140 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_105 = self.L__mod___transformer_h_17_mlp_fc_out(hidden_states_104);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_17 = self.L__mod___transformer_h_17_mlp_dropout(hidden_states_105);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_106 = attn_output_106 + feed_forward_hidden_states_17;  attn_output_106 = feed_forward_hidden_states_17 = None
    residual_18 = add_106 + residual_17;  add_106 = residual_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_108 = self.L__mod___transformer_h_18_ln_1(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_90 = self.L__mod___transformer_h_18_attn_q_proj(hidden_states_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_90 = self.L__mod___transformer_h_18_attn_k_proj(hidden_states_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_36 = self.L__mod___transformer_h_18_attn_v_proj(hidden_states_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_91 = query_90.view((1, 128, 16, 256));  query_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_91 = key_90.view((1, 128, 16, 256));  key_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_92 = value_36.view((1, 128, 16, 256));  value_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_37 = tensor_92.permute(0, 2, 1, 3);  tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_36 = self.L__mod___transformer_h_18_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_37 = embed_positions_36.repeat(1, 1, 1);  embed_positions_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_19 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_18 = unsqueeze_19.repeat(1, 1, 64);  unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_18 = torch.gather(embed_positions_37, 1, repeated_position_ids_18);  embed_positions_37 = repeated_position_ids_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_18 = torch.functional.split(sincos_18, 32, dim = -1);  sincos_18 = None
    sin_54 = split_18[0]
    cos_54 = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_36 = key_91[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_18 = key_91[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_36 = query_91[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_18 = query_91[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_276 = sin_54[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_55 = torch.repeat_interleave(getitem_276, 2, 3);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_277 = cos_54[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_55 = torch.repeat_interleave(getitem_277, 2, 3);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_144 = k_rot_36 * cos_55;  cos_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_36 = k_rot_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_36 = k_rot_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_36 = -x2_36;  x2_36 = None
    x_36 = torch.stack((neg_36, x1_36), dim = -1);  neg_36 = x1_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_36 = x_36.flatten(-2);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_145 = flatten_36 * sin_55;  flatten_36 = sin_55 = None
    k_rot_37 = mul_144 + mul_145;  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_280 = sin_54[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_54 = None
    sin_56 = torch.repeat_interleave(getitem_280, 2, 3);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_281 = cos_54[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_54 = None
    cos_56 = torch.repeat_interleave(getitem_281, 2, 3);  getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_146 = q_rot_36 * cos_56;  cos_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_37 = q_rot_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_37 = q_rot_36[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_37 = -x2_37;  x2_37 = None
    x_37 = torch.stack((neg_37, x1_37), dim = -1);  neg_37 = x1_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_37 = x_37.flatten(-2);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_147 = flatten_37 * sin_56;  flatten_37 = sin_56 = None
    q_rot_37 = mul_146 + mul_147;  mul_146 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_92 = torch.cat([k_rot_37, k_pass_18], dim = -1);  k_rot_37 = k_pass_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_92 = torch.cat([q_rot_37, q_pass_18], dim = -1);  q_rot_37 = q_pass_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_93 = key_92.permute(0, 2, 1, 3);  key_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_93 = query_92.permute(0, 2, 1, 3);  query_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_18_attn_bias = self.L__mod___transformer_h_18_attn_bias
    causal_mask_18 = l__mod___transformer_h_18_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_18_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_94 = query_93.to(torch.float32);  query_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_94 = key_93.to(torch.float32);  key_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_18 = key_94.transpose(-1, -2);  key_94 = None
    attn_weights_126 = torch.matmul(query_94, transpose_18);  query_94 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_93 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_18 = tensor_93.to(device(type='cpu'));  tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_127 = torch.where(causal_mask_18, attn_weights_126, mask_value_18);  causal_mask_18 = attn_weights_126 = mask_value_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_18_attn_scale_attn = self.L__mod___transformer_h_18_attn_scale_attn
    attn_weights_128 = attn_weights_127 / l__mod___transformer_h_18_attn_scale_attn;  attn_weights_127 = l__mod___transformer_h_18_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_129 = torch.nn.functional.softmax(attn_weights_128, dim = -1);  attn_weights_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_130 = attn_weights_129.to(torch.float32);  attn_weights_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_132 = self.L__mod___transformer_h_18_attn_attn_dropout(attn_weights_130);  attn_weights_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_108 = torch.matmul(attn_weights_132, value_37);  attn_weights_132 = value_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_75 = attn_output_108.permute(0, 2, 1, 3);  attn_output_108 = None
    tensor_94 = permute_75.contiguous();  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_109 = tensor_94.view((1, 128, 4096));  tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_110 = self.L__mod___transformer_h_18_attn_out_proj(attn_output_109);  attn_output_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_112 = self.L__mod___transformer_h_18_attn_resid_dropout(attn_output_110);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_109 = self.L__mod___transformer_h_18_mlp_fc_in(hidden_states_108);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_148 = 0.5 * hidden_states_109
    pow_19 = torch.pow(hidden_states_109, 3.0)
    mul_149 = 0.044715 * pow_19;  pow_19 = None
    add_110 = hidden_states_109 + mul_149;  hidden_states_109 = mul_149 = None
    mul_150 = 0.7978845608028654 * add_110;  add_110 = None
    tanh_18 = torch.tanh(mul_150);  mul_150 = None
    add_111 = 1.0 + tanh_18;  tanh_18 = None
    hidden_states_110 = mul_148 * add_111;  mul_148 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_111 = self.L__mod___transformer_h_18_mlp_fc_out(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_18 = self.L__mod___transformer_h_18_mlp_dropout(hidden_states_111);  hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_112 = attn_output_112 + feed_forward_hidden_states_18;  attn_output_112 = feed_forward_hidden_states_18 = None
    residual_19 = add_112 + residual_18;  add_112 = residual_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_114 = self.L__mod___transformer_h_19_ln_1(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_95 = self.L__mod___transformer_h_19_attn_q_proj(hidden_states_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_95 = self.L__mod___transformer_h_19_attn_k_proj(hidden_states_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_38 = self.L__mod___transformer_h_19_attn_v_proj(hidden_states_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_96 = query_95.view((1, 128, 16, 256));  query_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_96 = key_95.view((1, 128, 16, 256));  key_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_97 = value_38.view((1, 128, 16, 256));  value_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_39 = tensor_97.permute(0, 2, 1, 3);  tensor_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_38 = self.L__mod___transformer_h_19_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_39 = embed_positions_38.repeat(1, 1, 1);  embed_positions_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_20 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_19 = unsqueeze_20.repeat(1, 1, 64);  unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_19 = torch.gather(embed_positions_39, 1, repeated_position_ids_19);  embed_positions_39 = repeated_position_ids_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_19 = torch.functional.split(sincos_19, 32, dim = -1);  sincos_19 = None
    sin_57 = split_19[0]
    cos_57 = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_38 = key_96[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_19 = key_96[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_38 = query_96[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_19 = query_96[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_291 = sin_57[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_58 = torch.repeat_interleave(getitem_291, 2, 3);  getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_292 = cos_57[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_58 = torch.repeat_interleave(getitem_292, 2, 3);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_152 = k_rot_38 * cos_58;  cos_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_38 = k_rot_38[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_38 = k_rot_38[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_38 = -x2_38;  x2_38 = None
    x_38 = torch.stack((neg_38, x1_38), dim = -1);  neg_38 = x1_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_38 = x_38.flatten(-2);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_153 = flatten_38 * sin_58;  flatten_38 = sin_58 = None
    k_rot_39 = mul_152 + mul_153;  mul_152 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_295 = sin_57[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_57 = None
    sin_59 = torch.repeat_interleave(getitem_295, 2, 3);  getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_296 = cos_57[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_57 = None
    cos_59 = torch.repeat_interleave(getitem_296, 2, 3);  getitem_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_154 = q_rot_38 * cos_59;  cos_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_39 = q_rot_38[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_39 = q_rot_38[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_39 = -x2_39;  x2_39 = None
    x_39 = torch.stack((neg_39, x1_39), dim = -1);  neg_39 = x1_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_39 = x_39.flatten(-2);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_155 = flatten_39 * sin_59;  flatten_39 = sin_59 = None
    q_rot_39 = mul_154 + mul_155;  mul_154 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_97 = torch.cat([k_rot_39, k_pass_19], dim = -1);  k_rot_39 = k_pass_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_97 = torch.cat([q_rot_39, q_pass_19], dim = -1);  q_rot_39 = q_pass_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_98 = key_97.permute(0, 2, 1, 3);  key_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_98 = query_97.permute(0, 2, 1, 3);  query_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_19_attn_bias = self.L__mod___transformer_h_19_attn_bias
    causal_mask_19 = l__mod___transformer_h_19_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_19_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_99 = query_98.to(torch.float32);  query_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_99 = key_98.to(torch.float32);  key_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_19 = key_99.transpose(-1, -2);  key_99 = None
    attn_weights_133 = torch.matmul(query_99, transpose_19);  query_99 = transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_98 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_19 = tensor_98.to(device(type='cpu'));  tensor_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_134 = torch.where(causal_mask_19, attn_weights_133, mask_value_19);  causal_mask_19 = attn_weights_133 = mask_value_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_19_attn_scale_attn = self.L__mod___transformer_h_19_attn_scale_attn
    attn_weights_135 = attn_weights_134 / l__mod___transformer_h_19_attn_scale_attn;  attn_weights_134 = l__mod___transformer_h_19_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_136 = torch.nn.functional.softmax(attn_weights_135, dim = -1);  attn_weights_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_137 = attn_weights_136.to(torch.float32);  attn_weights_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_139 = self.L__mod___transformer_h_19_attn_attn_dropout(attn_weights_137);  attn_weights_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_114 = torch.matmul(attn_weights_139, value_39);  attn_weights_139 = value_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_79 = attn_output_114.permute(0, 2, 1, 3);  attn_output_114 = None
    tensor_99 = permute_79.contiguous();  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_115 = tensor_99.view((1, 128, 4096));  tensor_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_116 = self.L__mod___transformer_h_19_attn_out_proj(attn_output_115);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_118 = self.L__mod___transformer_h_19_attn_resid_dropout(attn_output_116);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_115 = self.L__mod___transformer_h_19_mlp_fc_in(hidden_states_114);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156 = 0.5 * hidden_states_115
    pow_20 = torch.pow(hidden_states_115, 3.0)
    mul_157 = 0.044715 * pow_20;  pow_20 = None
    add_116 = hidden_states_115 + mul_157;  hidden_states_115 = mul_157 = None
    mul_158 = 0.7978845608028654 * add_116;  add_116 = None
    tanh_19 = torch.tanh(mul_158);  mul_158 = None
    add_117 = 1.0 + tanh_19;  tanh_19 = None
    hidden_states_116 = mul_156 * add_117;  mul_156 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_117 = self.L__mod___transformer_h_19_mlp_fc_out(hidden_states_116);  hidden_states_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_19 = self.L__mod___transformer_h_19_mlp_dropout(hidden_states_117);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_118 = attn_output_118 + feed_forward_hidden_states_19;  attn_output_118 = feed_forward_hidden_states_19 = None
    residual_20 = add_118 + residual_19;  add_118 = residual_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_120 = self.L__mod___transformer_h_20_ln_1(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_100 = self.L__mod___transformer_h_20_attn_q_proj(hidden_states_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_100 = self.L__mod___transformer_h_20_attn_k_proj(hidden_states_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_40 = self.L__mod___transformer_h_20_attn_v_proj(hidden_states_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_101 = query_100.view((1, 128, 16, 256));  query_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_101 = key_100.view((1, 128, 16, 256));  key_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_102 = value_40.view((1, 128, 16, 256));  value_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_41 = tensor_102.permute(0, 2, 1, 3);  tensor_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_40 = self.L__mod___transformer_h_20_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_41 = embed_positions_40.repeat(1, 1, 1);  embed_positions_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_21 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_20 = unsqueeze_21.repeat(1, 1, 64);  unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_20 = torch.gather(embed_positions_41, 1, repeated_position_ids_20);  embed_positions_41 = repeated_position_ids_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_20 = torch.functional.split(sincos_20, 32, dim = -1);  sincos_20 = None
    sin_60 = split_20[0]
    cos_60 = split_20[1];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_40 = key_101[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_20 = key_101[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_40 = query_101[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_20 = query_101[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_306 = sin_60[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_61 = torch.repeat_interleave(getitem_306, 2, 3);  getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_307 = cos_60[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_61 = torch.repeat_interleave(getitem_307, 2, 3);  getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_160 = k_rot_40 * cos_61;  cos_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_40 = k_rot_40[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_40 = k_rot_40[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_40 = -x2_40;  x2_40 = None
    x_40 = torch.stack((neg_40, x1_40), dim = -1);  neg_40 = x1_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_40 = x_40.flatten(-2);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_161 = flatten_40 * sin_61;  flatten_40 = sin_61 = None
    k_rot_41 = mul_160 + mul_161;  mul_160 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_310 = sin_60[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_60 = None
    sin_62 = torch.repeat_interleave(getitem_310, 2, 3);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_311 = cos_60[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_60 = None
    cos_62 = torch.repeat_interleave(getitem_311, 2, 3);  getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_162 = q_rot_40 * cos_62;  cos_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_41 = q_rot_40[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_41 = q_rot_40[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_41 = -x2_41;  x2_41 = None
    x_41 = torch.stack((neg_41, x1_41), dim = -1);  neg_41 = x1_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_41 = x_41.flatten(-2);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_163 = flatten_41 * sin_62;  flatten_41 = sin_62 = None
    q_rot_41 = mul_162 + mul_163;  mul_162 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_102 = torch.cat([k_rot_41, k_pass_20], dim = -1);  k_rot_41 = k_pass_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_102 = torch.cat([q_rot_41, q_pass_20], dim = -1);  q_rot_41 = q_pass_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_103 = key_102.permute(0, 2, 1, 3);  key_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_103 = query_102.permute(0, 2, 1, 3);  query_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_20_attn_bias = self.L__mod___transformer_h_20_attn_bias
    causal_mask_20 = l__mod___transformer_h_20_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_20_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_104 = query_103.to(torch.float32);  query_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_104 = key_103.to(torch.float32);  key_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_20 = key_104.transpose(-1, -2);  key_104 = None
    attn_weights_140 = torch.matmul(query_104, transpose_20);  query_104 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_103 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_20 = tensor_103.to(device(type='cpu'));  tensor_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_141 = torch.where(causal_mask_20, attn_weights_140, mask_value_20);  causal_mask_20 = attn_weights_140 = mask_value_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_20_attn_scale_attn = self.L__mod___transformer_h_20_attn_scale_attn
    attn_weights_142 = attn_weights_141 / l__mod___transformer_h_20_attn_scale_attn;  attn_weights_141 = l__mod___transformer_h_20_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_143 = torch.nn.functional.softmax(attn_weights_142, dim = -1);  attn_weights_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_144 = attn_weights_143.to(torch.float32);  attn_weights_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_146 = self.L__mod___transformer_h_20_attn_attn_dropout(attn_weights_144);  attn_weights_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_120 = torch.matmul(attn_weights_146, value_41);  attn_weights_146 = value_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_83 = attn_output_120.permute(0, 2, 1, 3);  attn_output_120 = None
    tensor_104 = permute_83.contiguous();  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_121 = tensor_104.view((1, 128, 4096));  tensor_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_122 = self.L__mod___transformer_h_20_attn_out_proj(attn_output_121);  attn_output_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_124 = self.L__mod___transformer_h_20_attn_resid_dropout(attn_output_122);  attn_output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_121 = self.L__mod___transformer_h_20_mlp_fc_in(hidden_states_120);  hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164 = 0.5 * hidden_states_121
    pow_21 = torch.pow(hidden_states_121, 3.0)
    mul_165 = 0.044715 * pow_21;  pow_21 = None
    add_122 = hidden_states_121 + mul_165;  hidden_states_121 = mul_165 = None
    mul_166 = 0.7978845608028654 * add_122;  add_122 = None
    tanh_20 = torch.tanh(mul_166);  mul_166 = None
    add_123 = 1.0 + tanh_20;  tanh_20 = None
    hidden_states_122 = mul_164 * add_123;  mul_164 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_123 = self.L__mod___transformer_h_20_mlp_fc_out(hidden_states_122);  hidden_states_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_20 = self.L__mod___transformer_h_20_mlp_dropout(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_124 = attn_output_124 + feed_forward_hidden_states_20;  attn_output_124 = feed_forward_hidden_states_20 = None
    residual_21 = add_124 + residual_20;  add_124 = residual_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_126 = self.L__mod___transformer_h_21_ln_1(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_105 = self.L__mod___transformer_h_21_attn_q_proj(hidden_states_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_105 = self.L__mod___transformer_h_21_attn_k_proj(hidden_states_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_42 = self.L__mod___transformer_h_21_attn_v_proj(hidden_states_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_106 = query_105.view((1, 128, 16, 256));  query_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_106 = key_105.view((1, 128, 16, 256));  key_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_107 = value_42.view((1, 128, 16, 256));  value_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_43 = tensor_107.permute(0, 2, 1, 3);  tensor_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_42 = self.L__mod___transformer_h_21_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_43 = embed_positions_42.repeat(1, 1, 1);  embed_positions_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_22 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_21 = unsqueeze_22.repeat(1, 1, 64);  unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_21 = torch.gather(embed_positions_43, 1, repeated_position_ids_21);  embed_positions_43 = repeated_position_ids_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_21 = torch.functional.split(sincos_21, 32, dim = -1);  sincos_21 = None
    sin_63 = split_21[0]
    cos_63 = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_42 = key_106[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_21 = key_106[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_42 = query_106[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_21 = query_106[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_321 = sin_63[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_64 = torch.repeat_interleave(getitem_321, 2, 3);  getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_322 = cos_63[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_64 = torch.repeat_interleave(getitem_322, 2, 3);  getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_168 = k_rot_42 * cos_64;  cos_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_42 = k_rot_42[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_42 = k_rot_42[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_42 = -x2_42;  x2_42 = None
    x_42 = torch.stack((neg_42, x1_42), dim = -1);  neg_42 = x1_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_42 = x_42.flatten(-2);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_169 = flatten_42 * sin_64;  flatten_42 = sin_64 = None
    k_rot_43 = mul_168 + mul_169;  mul_168 = mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_325 = sin_63[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_63 = None
    sin_65 = torch.repeat_interleave(getitem_325, 2, 3);  getitem_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_326 = cos_63[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_63 = None
    cos_65 = torch.repeat_interleave(getitem_326, 2, 3);  getitem_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_170 = q_rot_42 * cos_65;  cos_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_43 = q_rot_42[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_43 = q_rot_42[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_43 = -x2_43;  x2_43 = None
    x_43 = torch.stack((neg_43, x1_43), dim = -1);  neg_43 = x1_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_43 = x_43.flatten(-2);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_171 = flatten_43 * sin_65;  flatten_43 = sin_65 = None
    q_rot_43 = mul_170 + mul_171;  mul_170 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_107 = torch.cat([k_rot_43, k_pass_21], dim = -1);  k_rot_43 = k_pass_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_107 = torch.cat([q_rot_43, q_pass_21], dim = -1);  q_rot_43 = q_pass_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_108 = key_107.permute(0, 2, 1, 3);  key_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_108 = query_107.permute(0, 2, 1, 3);  query_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_21_attn_bias = self.L__mod___transformer_h_21_attn_bias
    causal_mask_21 = l__mod___transformer_h_21_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_21_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_109 = query_108.to(torch.float32);  query_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_109 = key_108.to(torch.float32);  key_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_21 = key_109.transpose(-1, -2);  key_109 = None
    attn_weights_147 = torch.matmul(query_109, transpose_21);  query_109 = transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_108 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_21 = tensor_108.to(device(type='cpu'));  tensor_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_148 = torch.where(causal_mask_21, attn_weights_147, mask_value_21);  causal_mask_21 = attn_weights_147 = mask_value_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_21_attn_scale_attn = self.L__mod___transformer_h_21_attn_scale_attn
    attn_weights_149 = attn_weights_148 / l__mod___transformer_h_21_attn_scale_attn;  attn_weights_148 = l__mod___transformer_h_21_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_150 = torch.nn.functional.softmax(attn_weights_149, dim = -1);  attn_weights_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_151 = attn_weights_150.to(torch.float32);  attn_weights_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_153 = self.L__mod___transformer_h_21_attn_attn_dropout(attn_weights_151);  attn_weights_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_126 = torch.matmul(attn_weights_153, value_43);  attn_weights_153 = value_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_87 = attn_output_126.permute(0, 2, 1, 3);  attn_output_126 = None
    tensor_109 = permute_87.contiguous();  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_127 = tensor_109.view((1, 128, 4096));  tensor_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_128 = self.L__mod___transformer_h_21_attn_out_proj(attn_output_127);  attn_output_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_130 = self.L__mod___transformer_h_21_attn_resid_dropout(attn_output_128);  attn_output_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_127 = self.L__mod___transformer_h_21_mlp_fc_in(hidden_states_126);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_172 = 0.5 * hidden_states_127
    pow_22 = torch.pow(hidden_states_127, 3.0)
    mul_173 = 0.044715 * pow_22;  pow_22 = None
    add_128 = hidden_states_127 + mul_173;  hidden_states_127 = mul_173 = None
    mul_174 = 0.7978845608028654 * add_128;  add_128 = None
    tanh_21 = torch.tanh(mul_174);  mul_174 = None
    add_129 = 1.0 + tanh_21;  tanh_21 = None
    hidden_states_128 = mul_172 * add_129;  mul_172 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_129 = self.L__mod___transformer_h_21_mlp_fc_out(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_21 = self.L__mod___transformer_h_21_mlp_dropout(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_130 = attn_output_130 + feed_forward_hidden_states_21;  attn_output_130 = feed_forward_hidden_states_21 = None
    residual_22 = add_130 + residual_21;  add_130 = residual_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_132 = self.L__mod___transformer_h_22_ln_1(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_110 = self.L__mod___transformer_h_22_attn_q_proj(hidden_states_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_110 = self.L__mod___transformer_h_22_attn_k_proj(hidden_states_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_44 = self.L__mod___transformer_h_22_attn_v_proj(hidden_states_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_111 = query_110.view((1, 128, 16, 256));  query_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_111 = key_110.view((1, 128, 16, 256));  key_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_112 = value_44.view((1, 128, 16, 256));  value_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_45 = tensor_112.permute(0, 2, 1, 3);  tensor_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_44 = self.L__mod___transformer_h_22_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_45 = embed_positions_44.repeat(1, 1, 1);  embed_positions_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_23 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_22 = unsqueeze_23.repeat(1, 1, 64);  unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_22 = torch.gather(embed_positions_45, 1, repeated_position_ids_22);  embed_positions_45 = repeated_position_ids_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_22 = torch.functional.split(sincos_22, 32, dim = -1);  sincos_22 = None
    sin_66 = split_22[0]
    cos_66 = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_44 = key_111[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_22 = key_111[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_44 = query_111[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_22 = query_111[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_336 = sin_66[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_67 = torch.repeat_interleave(getitem_336, 2, 3);  getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_337 = cos_66[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_67 = torch.repeat_interleave(getitem_337, 2, 3);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_176 = k_rot_44 * cos_67;  cos_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_44 = k_rot_44[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_44 = k_rot_44[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_44 = -x2_44;  x2_44 = None
    x_44 = torch.stack((neg_44, x1_44), dim = -1);  neg_44 = x1_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_44 = x_44.flatten(-2);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_177 = flatten_44 * sin_67;  flatten_44 = sin_67 = None
    k_rot_45 = mul_176 + mul_177;  mul_176 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_340 = sin_66[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_66 = None
    sin_68 = torch.repeat_interleave(getitem_340, 2, 3);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_341 = cos_66[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_66 = None
    cos_68 = torch.repeat_interleave(getitem_341, 2, 3);  getitem_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_178 = q_rot_44 * cos_68;  cos_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_45 = q_rot_44[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_45 = q_rot_44[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_45 = -x2_45;  x2_45 = None
    x_45 = torch.stack((neg_45, x1_45), dim = -1);  neg_45 = x1_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_45 = x_45.flatten(-2);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_179 = flatten_45 * sin_68;  flatten_45 = sin_68 = None
    q_rot_45 = mul_178 + mul_179;  mul_178 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_112 = torch.cat([k_rot_45, k_pass_22], dim = -1);  k_rot_45 = k_pass_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_112 = torch.cat([q_rot_45, q_pass_22], dim = -1);  q_rot_45 = q_pass_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_113 = key_112.permute(0, 2, 1, 3);  key_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_113 = query_112.permute(0, 2, 1, 3);  query_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_22_attn_bias = self.L__mod___transformer_h_22_attn_bias
    causal_mask_22 = l__mod___transformer_h_22_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_22_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_114 = query_113.to(torch.float32);  query_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_114 = key_113.to(torch.float32);  key_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_22 = key_114.transpose(-1, -2);  key_114 = None
    attn_weights_154 = torch.matmul(query_114, transpose_22);  query_114 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_113 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_22 = tensor_113.to(device(type='cpu'));  tensor_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_155 = torch.where(causal_mask_22, attn_weights_154, mask_value_22);  causal_mask_22 = attn_weights_154 = mask_value_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_22_attn_scale_attn = self.L__mod___transformer_h_22_attn_scale_attn
    attn_weights_156 = attn_weights_155 / l__mod___transformer_h_22_attn_scale_attn;  attn_weights_155 = l__mod___transformer_h_22_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_157 = torch.nn.functional.softmax(attn_weights_156, dim = -1);  attn_weights_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_158 = attn_weights_157.to(torch.float32);  attn_weights_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_160 = self.L__mod___transformer_h_22_attn_attn_dropout(attn_weights_158);  attn_weights_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_132 = torch.matmul(attn_weights_160, value_45);  attn_weights_160 = value_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_91 = attn_output_132.permute(0, 2, 1, 3);  attn_output_132 = None
    tensor_114 = permute_91.contiguous();  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_133 = tensor_114.view((1, 128, 4096));  tensor_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_134 = self.L__mod___transformer_h_22_attn_out_proj(attn_output_133);  attn_output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_136 = self.L__mod___transformer_h_22_attn_resid_dropout(attn_output_134);  attn_output_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_133 = self.L__mod___transformer_h_22_mlp_fc_in(hidden_states_132);  hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_180 = 0.5 * hidden_states_133
    pow_23 = torch.pow(hidden_states_133, 3.0)
    mul_181 = 0.044715 * pow_23;  pow_23 = None
    add_134 = hidden_states_133 + mul_181;  hidden_states_133 = mul_181 = None
    mul_182 = 0.7978845608028654 * add_134;  add_134 = None
    tanh_22 = torch.tanh(mul_182);  mul_182 = None
    add_135 = 1.0 + tanh_22;  tanh_22 = None
    hidden_states_134 = mul_180 * add_135;  mul_180 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_135 = self.L__mod___transformer_h_22_mlp_fc_out(hidden_states_134);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_22 = self.L__mod___transformer_h_22_mlp_dropout(hidden_states_135);  hidden_states_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_136 = attn_output_136 + feed_forward_hidden_states_22;  attn_output_136 = feed_forward_hidden_states_22 = None
    residual_23 = add_136 + residual_22;  add_136 = residual_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_138 = self.L__mod___transformer_h_23_ln_1(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_115 = self.L__mod___transformer_h_23_attn_q_proj(hidden_states_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_115 = self.L__mod___transformer_h_23_attn_k_proj(hidden_states_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_46 = self.L__mod___transformer_h_23_attn_v_proj(hidden_states_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_116 = query_115.view((1, 128, 16, 256));  query_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_116 = key_115.view((1, 128, 16, 256));  key_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_117 = value_46.view((1, 128, 16, 256));  value_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_47 = tensor_117.permute(0, 2, 1, 3);  tensor_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_46 = self.L__mod___transformer_h_23_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_47 = embed_positions_46.repeat(1, 1, 1);  embed_positions_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_24 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_23 = unsqueeze_24.repeat(1, 1, 64);  unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_23 = torch.gather(embed_positions_47, 1, repeated_position_ids_23);  embed_positions_47 = repeated_position_ids_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_23 = torch.functional.split(sincos_23, 32, dim = -1);  sincos_23 = None
    sin_69 = split_23[0]
    cos_69 = split_23[1];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_46 = key_116[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_23 = key_116[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_46 = query_116[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_23 = query_116[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_351 = sin_69[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_70 = torch.repeat_interleave(getitem_351, 2, 3);  getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_352 = cos_69[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_70 = torch.repeat_interleave(getitem_352, 2, 3);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_184 = k_rot_46 * cos_70;  cos_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_46 = k_rot_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_46 = k_rot_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_46 = -x2_46;  x2_46 = None
    x_46 = torch.stack((neg_46, x1_46), dim = -1);  neg_46 = x1_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_46 = x_46.flatten(-2);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_185 = flatten_46 * sin_70;  flatten_46 = sin_70 = None
    k_rot_47 = mul_184 + mul_185;  mul_184 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_355 = sin_69[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_69 = None
    sin_71 = torch.repeat_interleave(getitem_355, 2, 3);  getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_356 = cos_69[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_69 = None
    cos_71 = torch.repeat_interleave(getitem_356, 2, 3);  getitem_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_186 = q_rot_46 * cos_71;  cos_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_47 = q_rot_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_47 = q_rot_46[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_47 = -x2_47;  x2_47 = None
    x_47 = torch.stack((neg_47, x1_47), dim = -1);  neg_47 = x1_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_47 = x_47.flatten(-2);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_187 = flatten_47 * sin_71;  flatten_47 = sin_71 = None
    q_rot_47 = mul_186 + mul_187;  mul_186 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_117 = torch.cat([k_rot_47, k_pass_23], dim = -1);  k_rot_47 = k_pass_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_117 = torch.cat([q_rot_47, q_pass_23], dim = -1);  q_rot_47 = q_pass_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_118 = key_117.permute(0, 2, 1, 3);  key_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_118 = query_117.permute(0, 2, 1, 3);  query_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_23_attn_bias = self.L__mod___transformer_h_23_attn_bias
    causal_mask_23 = l__mod___transformer_h_23_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_23_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_119 = query_118.to(torch.float32);  query_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_119 = key_118.to(torch.float32);  key_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_23 = key_119.transpose(-1, -2);  key_119 = None
    attn_weights_161 = torch.matmul(query_119, transpose_23);  query_119 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_118 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_23 = tensor_118.to(device(type='cpu'));  tensor_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_162 = torch.where(causal_mask_23, attn_weights_161, mask_value_23);  causal_mask_23 = attn_weights_161 = mask_value_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_23_attn_scale_attn = self.L__mod___transformer_h_23_attn_scale_attn
    attn_weights_163 = attn_weights_162 / l__mod___transformer_h_23_attn_scale_attn;  attn_weights_162 = l__mod___transformer_h_23_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_164 = torch.nn.functional.softmax(attn_weights_163, dim = -1);  attn_weights_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_165 = attn_weights_164.to(torch.float32);  attn_weights_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_167 = self.L__mod___transformer_h_23_attn_attn_dropout(attn_weights_165);  attn_weights_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_138 = torch.matmul(attn_weights_167, value_47);  attn_weights_167 = value_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95 = attn_output_138.permute(0, 2, 1, 3);  attn_output_138 = None
    tensor_119 = permute_95.contiguous();  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_139 = tensor_119.view((1, 128, 4096));  tensor_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_140 = self.L__mod___transformer_h_23_attn_out_proj(attn_output_139);  attn_output_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_142 = self.L__mod___transformer_h_23_attn_resid_dropout(attn_output_140);  attn_output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_139 = self.L__mod___transformer_h_23_mlp_fc_in(hidden_states_138);  hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_188 = 0.5 * hidden_states_139
    pow_24 = torch.pow(hidden_states_139, 3.0)
    mul_189 = 0.044715 * pow_24;  pow_24 = None
    add_140 = hidden_states_139 + mul_189;  hidden_states_139 = mul_189 = None
    mul_190 = 0.7978845608028654 * add_140;  add_140 = None
    tanh_23 = torch.tanh(mul_190);  mul_190 = None
    add_141 = 1.0 + tanh_23;  tanh_23 = None
    hidden_states_140 = mul_188 * add_141;  mul_188 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_141 = self.L__mod___transformer_h_23_mlp_fc_out(hidden_states_140);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_23 = self.L__mod___transformer_h_23_mlp_dropout(hidden_states_141);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_142 = attn_output_142 + feed_forward_hidden_states_23;  attn_output_142 = feed_forward_hidden_states_23 = None
    residual_24 = add_142 + residual_23;  add_142 = residual_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_144 = self.L__mod___transformer_h_24_ln_1(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_120 = self.L__mod___transformer_h_24_attn_q_proj(hidden_states_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_120 = self.L__mod___transformer_h_24_attn_k_proj(hidden_states_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_48 = self.L__mod___transformer_h_24_attn_v_proj(hidden_states_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_121 = query_120.view((1, 128, 16, 256));  query_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_121 = key_120.view((1, 128, 16, 256));  key_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_122 = value_48.view((1, 128, 16, 256));  value_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_49 = tensor_122.permute(0, 2, 1, 3);  tensor_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_48 = self.L__mod___transformer_h_24_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_49 = embed_positions_48.repeat(1, 1, 1);  embed_positions_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_25 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_24 = unsqueeze_25.repeat(1, 1, 64);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_24 = torch.gather(embed_positions_49, 1, repeated_position_ids_24);  embed_positions_49 = repeated_position_ids_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_24 = torch.functional.split(sincos_24, 32, dim = -1);  sincos_24 = None
    sin_72 = split_24[0]
    cos_72 = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_48 = key_121[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_24 = key_121[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_48 = query_121[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_24 = query_121[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_366 = sin_72[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_73 = torch.repeat_interleave(getitem_366, 2, 3);  getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_367 = cos_72[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_73 = torch.repeat_interleave(getitem_367, 2, 3);  getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_192 = k_rot_48 * cos_73;  cos_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_48 = k_rot_48[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_48 = k_rot_48[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_48 = -x2_48;  x2_48 = None
    x_48 = torch.stack((neg_48, x1_48), dim = -1);  neg_48 = x1_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_48 = x_48.flatten(-2);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_193 = flatten_48 * sin_73;  flatten_48 = sin_73 = None
    k_rot_49 = mul_192 + mul_193;  mul_192 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_370 = sin_72[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_72 = None
    sin_74 = torch.repeat_interleave(getitem_370, 2, 3);  getitem_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_371 = cos_72[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_72 = None
    cos_74 = torch.repeat_interleave(getitem_371, 2, 3);  getitem_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_194 = q_rot_48 * cos_74;  cos_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_49 = q_rot_48[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_49 = q_rot_48[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_49 = -x2_49;  x2_49 = None
    x_49 = torch.stack((neg_49, x1_49), dim = -1);  neg_49 = x1_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_49 = x_49.flatten(-2);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_195 = flatten_49 * sin_74;  flatten_49 = sin_74 = None
    q_rot_49 = mul_194 + mul_195;  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_122 = torch.cat([k_rot_49, k_pass_24], dim = -1);  k_rot_49 = k_pass_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_122 = torch.cat([q_rot_49, q_pass_24], dim = -1);  q_rot_49 = q_pass_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_123 = key_122.permute(0, 2, 1, 3);  key_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_123 = query_122.permute(0, 2, 1, 3);  query_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_24_attn_bias = self.L__mod___transformer_h_24_attn_bias
    causal_mask_24 = l__mod___transformer_h_24_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_24_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_124 = query_123.to(torch.float32);  query_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_124 = key_123.to(torch.float32);  key_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_24 = key_124.transpose(-1, -2);  key_124 = None
    attn_weights_168 = torch.matmul(query_124, transpose_24);  query_124 = transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_123 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_24 = tensor_123.to(device(type='cpu'));  tensor_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_169 = torch.where(causal_mask_24, attn_weights_168, mask_value_24);  causal_mask_24 = attn_weights_168 = mask_value_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_24_attn_scale_attn = self.L__mod___transformer_h_24_attn_scale_attn
    attn_weights_170 = attn_weights_169 / l__mod___transformer_h_24_attn_scale_attn;  attn_weights_169 = l__mod___transformer_h_24_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_171 = torch.nn.functional.softmax(attn_weights_170, dim = -1);  attn_weights_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_172 = attn_weights_171.to(torch.float32);  attn_weights_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_174 = self.L__mod___transformer_h_24_attn_attn_dropout(attn_weights_172);  attn_weights_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_144 = torch.matmul(attn_weights_174, value_49);  attn_weights_174 = value_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_99 = attn_output_144.permute(0, 2, 1, 3);  attn_output_144 = None
    tensor_124 = permute_99.contiguous();  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_145 = tensor_124.view((1, 128, 4096));  tensor_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_146 = self.L__mod___transformer_h_24_attn_out_proj(attn_output_145);  attn_output_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_148 = self.L__mod___transformer_h_24_attn_resid_dropout(attn_output_146);  attn_output_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_145 = self.L__mod___transformer_h_24_mlp_fc_in(hidden_states_144);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_196 = 0.5 * hidden_states_145
    pow_25 = torch.pow(hidden_states_145, 3.0)
    mul_197 = 0.044715 * pow_25;  pow_25 = None
    add_146 = hidden_states_145 + mul_197;  hidden_states_145 = mul_197 = None
    mul_198 = 0.7978845608028654 * add_146;  add_146 = None
    tanh_24 = torch.tanh(mul_198);  mul_198 = None
    add_147 = 1.0 + tanh_24;  tanh_24 = None
    hidden_states_146 = mul_196 * add_147;  mul_196 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_147 = self.L__mod___transformer_h_24_mlp_fc_out(hidden_states_146);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_24 = self.L__mod___transformer_h_24_mlp_dropout(hidden_states_147);  hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_148 = attn_output_148 + feed_forward_hidden_states_24;  attn_output_148 = feed_forward_hidden_states_24 = None
    residual_25 = add_148 + residual_24;  add_148 = residual_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_150 = self.L__mod___transformer_h_25_ln_1(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_125 = self.L__mod___transformer_h_25_attn_q_proj(hidden_states_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_125 = self.L__mod___transformer_h_25_attn_k_proj(hidden_states_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_50 = self.L__mod___transformer_h_25_attn_v_proj(hidden_states_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_126 = query_125.view((1, 128, 16, 256));  query_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_126 = key_125.view((1, 128, 16, 256));  key_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_127 = value_50.view((1, 128, 16, 256));  value_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_51 = tensor_127.permute(0, 2, 1, 3);  tensor_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_50 = self.L__mod___transformer_h_25_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_51 = embed_positions_50.repeat(1, 1, 1);  embed_positions_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_26 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_25 = unsqueeze_26.repeat(1, 1, 64);  unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_25 = torch.gather(embed_positions_51, 1, repeated_position_ids_25);  embed_positions_51 = repeated_position_ids_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_25 = torch.functional.split(sincos_25, 32, dim = -1);  sincos_25 = None
    sin_75 = split_25[0]
    cos_75 = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_50 = key_126[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_25 = key_126[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_50 = query_126[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_25 = query_126[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_381 = sin_75[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_76 = torch.repeat_interleave(getitem_381, 2, 3);  getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_382 = cos_75[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_76 = torch.repeat_interleave(getitem_382, 2, 3);  getitem_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_200 = k_rot_50 * cos_76;  cos_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_50 = k_rot_50[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_50 = k_rot_50[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_50 = -x2_50;  x2_50 = None
    x_50 = torch.stack((neg_50, x1_50), dim = -1);  neg_50 = x1_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_50 = x_50.flatten(-2);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_201 = flatten_50 * sin_76;  flatten_50 = sin_76 = None
    k_rot_51 = mul_200 + mul_201;  mul_200 = mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_385 = sin_75[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_75 = None
    sin_77 = torch.repeat_interleave(getitem_385, 2, 3);  getitem_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_386 = cos_75[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_75 = None
    cos_77 = torch.repeat_interleave(getitem_386, 2, 3);  getitem_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_202 = q_rot_50 * cos_77;  cos_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_51 = q_rot_50[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_51 = q_rot_50[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_51 = -x2_51;  x2_51 = None
    x_51 = torch.stack((neg_51, x1_51), dim = -1);  neg_51 = x1_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_51 = x_51.flatten(-2);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_203 = flatten_51 * sin_77;  flatten_51 = sin_77 = None
    q_rot_51 = mul_202 + mul_203;  mul_202 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_127 = torch.cat([k_rot_51, k_pass_25], dim = -1);  k_rot_51 = k_pass_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_127 = torch.cat([q_rot_51, q_pass_25], dim = -1);  q_rot_51 = q_pass_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_128 = key_127.permute(0, 2, 1, 3);  key_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_128 = query_127.permute(0, 2, 1, 3);  query_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_25_attn_bias = self.L__mod___transformer_h_25_attn_bias
    causal_mask_25 = l__mod___transformer_h_25_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_25_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_129 = query_128.to(torch.float32);  query_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_129 = key_128.to(torch.float32);  key_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_25 = key_129.transpose(-1, -2);  key_129 = None
    attn_weights_175 = torch.matmul(query_129, transpose_25);  query_129 = transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_128 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_25 = tensor_128.to(device(type='cpu'));  tensor_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_176 = torch.where(causal_mask_25, attn_weights_175, mask_value_25);  causal_mask_25 = attn_weights_175 = mask_value_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_25_attn_scale_attn = self.L__mod___transformer_h_25_attn_scale_attn
    attn_weights_177 = attn_weights_176 / l__mod___transformer_h_25_attn_scale_attn;  attn_weights_176 = l__mod___transformer_h_25_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_178 = torch.nn.functional.softmax(attn_weights_177, dim = -1);  attn_weights_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_179 = attn_weights_178.to(torch.float32);  attn_weights_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_181 = self.L__mod___transformer_h_25_attn_attn_dropout(attn_weights_179);  attn_weights_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_150 = torch.matmul(attn_weights_181, value_51);  attn_weights_181 = value_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_103 = attn_output_150.permute(0, 2, 1, 3);  attn_output_150 = None
    tensor_129 = permute_103.contiguous();  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_151 = tensor_129.view((1, 128, 4096));  tensor_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_152 = self.L__mod___transformer_h_25_attn_out_proj(attn_output_151);  attn_output_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_154 = self.L__mod___transformer_h_25_attn_resid_dropout(attn_output_152);  attn_output_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_151 = self.L__mod___transformer_h_25_mlp_fc_in(hidden_states_150);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_204 = 0.5 * hidden_states_151
    pow_26 = torch.pow(hidden_states_151, 3.0)
    mul_205 = 0.044715 * pow_26;  pow_26 = None
    add_152 = hidden_states_151 + mul_205;  hidden_states_151 = mul_205 = None
    mul_206 = 0.7978845608028654 * add_152;  add_152 = None
    tanh_25 = torch.tanh(mul_206);  mul_206 = None
    add_153 = 1.0 + tanh_25;  tanh_25 = None
    hidden_states_152 = mul_204 * add_153;  mul_204 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_153 = self.L__mod___transformer_h_25_mlp_fc_out(hidden_states_152);  hidden_states_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_25 = self.L__mod___transformer_h_25_mlp_dropout(hidden_states_153);  hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_154 = attn_output_154 + feed_forward_hidden_states_25;  attn_output_154 = feed_forward_hidden_states_25 = None
    residual_26 = add_154 + residual_25;  add_154 = residual_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_156 = self.L__mod___transformer_h_26_ln_1(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_130 = self.L__mod___transformer_h_26_attn_q_proj(hidden_states_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_130 = self.L__mod___transformer_h_26_attn_k_proj(hidden_states_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_52 = self.L__mod___transformer_h_26_attn_v_proj(hidden_states_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_131 = query_130.view((1, 128, 16, 256));  query_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_131 = key_130.view((1, 128, 16, 256));  key_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_132 = value_52.view((1, 128, 16, 256));  value_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_53 = tensor_132.permute(0, 2, 1, 3);  tensor_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_52 = self.L__mod___transformer_h_26_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_53 = embed_positions_52.repeat(1, 1, 1);  embed_positions_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_27 = position_ids_1.unsqueeze(-1)
    repeated_position_ids_26 = unsqueeze_27.repeat(1, 1, 64);  unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_26 = torch.gather(embed_positions_53, 1, repeated_position_ids_26);  embed_positions_53 = repeated_position_ids_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_26 = torch.functional.split(sincos_26, 32, dim = -1);  sincos_26 = None
    sin_78 = split_26[0]
    cos_78 = split_26[1];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_52 = key_131[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_26 = key_131[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_52 = query_131[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_26 = query_131[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_396 = sin_78[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_79 = torch.repeat_interleave(getitem_396, 2, 3);  getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_397 = cos_78[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_79 = torch.repeat_interleave(getitem_397, 2, 3);  getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_208 = k_rot_52 * cos_79;  cos_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_52 = k_rot_52[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_52 = k_rot_52[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_52 = -x2_52;  x2_52 = None
    x_52 = torch.stack((neg_52, x1_52), dim = -1);  neg_52 = x1_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_52 = x_52.flatten(-2);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_209 = flatten_52 * sin_79;  flatten_52 = sin_79 = None
    k_rot_53 = mul_208 + mul_209;  mul_208 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_400 = sin_78[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_78 = None
    sin_80 = torch.repeat_interleave(getitem_400, 2, 3);  getitem_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_401 = cos_78[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_78 = None
    cos_80 = torch.repeat_interleave(getitem_401, 2, 3);  getitem_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_210 = q_rot_52 * cos_80;  cos_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_53 = q_rot_52[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_53 = q_rot_52[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_53 = -x2_53;  x2_53 = None
    x_53 = torch.stack((neg_53, x1_53), dim = -1);  neg_53 = x1_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_53 = x_53.flatten(-2);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_211 = flatten_53 * sin_80;  flatten_53 = sin_80 = None
    q_rot_53 = mul_210 + mul_211;  mul_210 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_132 = torch.cat([k_rot_53, k_pass_26], dim = -1);  k_rot_53 = k_pass_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_132 = torch.cat([q_rot_53, q_pass_26], dim = -1);  q_rot_53 = q_pass_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_133 = key_132.permute(0, 2, 1, 3);  key_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_133 = query_132.permute(0, 2, 1, 3);  query_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_26_attn_bias = self.L__mod___transformer_h_26_attn_bias
    causal_mask_26 = l__mod___transformer_h_26_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_26_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_134 = query_133.to(torch.float32);  query_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_134 = key_133.to(torch.float32);  key_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_26 = key_134.transpose(-1, -2);  key_134 = None
    attn_weights_182 = torch.matmul(query_134, transpose_26);  query_134 = transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_133 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_26 = tensor_133.to(device(type='cpu'));  tensor_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_183 = torch.where(causal_mask_26, attn_weights_182, mask_value_26);  causal_mask_26 = attn_weights_182 = mask_value_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_26_attn_scale_attn = self.L__mod___transformer_h_26_attn_scale_attn
    attn_weights_184 = attn_weights_183 / l__mod___transformer_h_26_attn_scale_attn;  attn_weights_183 = l__mod___transformer_h_26_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_185 = torch.nn.functional.softmax(attn_weights_184, dim = -1);  attn_weights_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_186 = attn_weights_185.to(torch.float32);  attn_weights_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_188 = self.L__mod___transformer_h_26_attn_attn_dropout(attn_weights_186);  attn_weights_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_156 = torch.matmul(attn_weights_188, value_53);  attn_weights_188 = value_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_107 = attn_output_156.permute(0, 2, 1, 3);  attn_output_156 = None
    tensor_134 = permute_107.contiguous();  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_157 = tensor_134.view((1, 128, 4096));  tensor_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_158 = self.L__mod___transformer_h_26_attn_out_proj(attn_output_157);  attn_output_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_160 = self.L__mod___transformer_h_26_attn_resid_dropout(attn_output_158);  attn_output_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_157 = self.L__mod___transformer_h_26_mlp_fc_in(hidden_states_156);  hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_212 = 0.5 * hidden_states_157
    pow_27 = torch.pow(hidden_states_157, 3.0)
    mul_213 = 0.044715 * pow_27;  pow_27 = None
    add_158 = hidden_states_157 + mul_213;  hidden_states_157 = mul_213 = None
    mul_214 = 0.7978845608028654 * add_158;  add_158 = None
    tanh_26 = torch.tanh(mul_214);  mul_214 = None
    add_159 = 1.0 + tanh_26;  tanh_26 = None
    hidden_states_158 = mul_212 * add_159;  mul_212 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_159 = self.L__mod___transformer_h_26_mlp_fc_out(hidden_states_158);  hidden_states_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_26 = self.L__mod___transformer_h_26_mlp_dropout(hidden_states_159);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_160 = attn_output_160 + feed_forward_hidden_states_26;  attn_output_160 = feed_forward_hidden_states_26 = None
    residual_27 = add_160 + residual_26;  add_160 = residual_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_162 = self.L__mod___transformer_h_27_ln_1(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    query_135 = self.L__mod___transformer_h_27_attn_q_proj(hidden_states_162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    key_135 = self.L__mod___transformer_h_27_attn_k_proj(hidden_states_162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    value_54 = self.L__mod___transformer_h_27_attn_v_proj(hidden_states_162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    query_136 = query_135.view((1, 128, 16, 256));  query_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    key_136 = key_135.view((1, 128, 16, 256));  key_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    tensor_137 = value_54.view((1, 128, 16, 256));  value_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_55 = tensor_137.permute(0, 2, 1, 3);  tensor_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:188, code: embed_positions = self.embed_positions
    embed_positions_54 = self.L__mod___transformer_h_27_attn_embed_positions
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    embed_positions_55 = embed_positions_54.repeat(1, 1, 1);  embed_positions_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_28 = position_ids_1.unsqueeze(-1);  position_ids_1 = None
    repeated_position_ids_27 = unsqueeze_28.repeat(1, 1, 64);  unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sincos_27 = torch.gather(embed_positions_55, 1, repeated_position_ids_27);  embed_positions_55 = repeated_position_ids_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_27 = torch.functional.split(sincos_27, 32, dim = -1);  sincos_27 = None
    sin_81 = split_27[0]
    cos_81 = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    k_rot_54 = key_136[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    k_pass_27 = key_136[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  key_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    q_rot_54 = query_136[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    q_pass_27 = query_136[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(64, None, None))];  query_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_411 = sin_81[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    sin_82 = torch.repeat_interleave(getitem_411, 2, 3);  getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_412 = cos_81[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))]
    cos_82 = torch.repeat_interleave(getitem_412, 2, 3);  getitem_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_216 = k_rot_54 * cos_82;  cos_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_54 = k_rot_54[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_54 = k_rot_54[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  k_rot_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_54 = -x2_54;  x2_54 = None
    x_54 = torch.stack((neg_54, x1_54), dim = -1);  neg_54 = x1_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_54 = x_54.flatten(-2);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_217 = flatten_54 * sin_82;  flatten_54 = sin_82 = None
    k_rot_55 = mul_216 + mul_217;  mul_216 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    getitem_415 = sin_81[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  sin_81 = None
    sin_83 = torch.repeat_interleave(getitem_415, 2, 3);  getitem_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    getitem_416 = cos_81[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  cos_81 = None
    cos_83 = torch.repeat_interleave(getitem_416, 2, 3);  getitem_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_218 = q_rot_54 * cos_83;  cos_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    x1_55 = q_rot_54[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, 2))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    x2_55 = q_rot_54[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  q_rot_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_55 = -x2_55;  x2_55 = None
    x_55 = torch.stack((neg_55, x1_55), dim = -1);  neg_55 = x1_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    flatten_55 = x_55.flatten(-2);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_219 = flatten_55 * sin_83;  flatten_55 = sin_83 = None
    q_rot_55 = mul_218 + mul_219;  mul_218 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    key_137 = torch.cat([k_rot_55, k_pass_27], dim = -1);  k_rot_55 = k_pass_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    query_137 = torch.cat([q_rot_55, q_pass_27], dim = -1);  q_rot_55 = q_pass_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    key_138 = key_137.permute(0, 2, 1, 3);  key_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    query_138 = query_137.permute(0, 2, 1, 3);  query_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_27_attn_bias = self.L__mod___transformer_h_27_attn_bias
    causal_mask_27 = l__mod___transformer_h_27_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_27_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:158, code: query = query.to(torch.float32)
    query_139 = query_138.to(torch.float32);  query_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:159, code: key = key.to(torch.float32)
    key_139 = key_138.to(torch.float32);  key_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_27 = key_139.transpose(-1, -2);  key_139 = None
    attn_weights_189 = torch.matmul(query_139, transpose_27);  query_139 = transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_138 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_27 = tensor_138.to(device(type='cpu'));  tensor_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_190 = torch.where(causal_mask_27, attn_weights_189, mask_value_27);  causal_mask_27 = attn_weights_189 = mask_value_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    l__mod___transformer_h_27_attn_scale_attn = self.L__mod___transformer_h_27_attn_scale_attn
    attn_weights_191 = attn_weights_190 / l__mod___transformer_h_27_attn_scale_attn;  attn_weights_190 = l__mod___transformer_h_27_attn_scale_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_192 = torch.nn.functional.softmax(attn_weights_191, dim = -1);  attn_weights_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:176, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_193 = attn_weights_192.to(torch.float32);  attn_weights_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_195 = self.L__mod___transformer_h_27_attn_attn_dropout(attn_weights_193);  attn_weights_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_162 = torch.matmul(attn_weights_195, value_55);  attn_weights_195 = value_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_111 = attn_output_162.permute(0, 2, 1, 3);  attn_output_162 = None
    tensor_139 = permute_111.contiguous();  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    attn_output_163 = tensor_139.view((1, 128, 4096));  tensor_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    attn_output_164 = self.L__mod___transformer_h_27_attn_out_proj(attn_output_163);  attn_output_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:261, code: attn_output = self.resid_dropout(attn_output)
    attn_output_166 = self.L__mod___transformer_h_27_attn_resid_dropout(attn_output_164);  attn_output_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    hidden_states_163 = self.L__mod___transformer_h_27_mlp_fc_in(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_220 = 0.5 * hidden_states_163
    pow_28 = torch.pow(hidden_states_163, 3.0)
    mul_221 = 0.044715 * pow_28;  pow_28 = None
    add_164 = hidden_states_163 + mul_221;  hidden_states_163 = mul_221 = None
    mul_222 = 0.7978845608028654 * add_164;  add_164 = None
    tanh_27 = torch.tanh(mul_222);  mul_222 = None
    add_165 = 1.0 + tanh_27;  tanh_27 = None
    hidden_states_164 = mul_220 * add_165;  mul_220 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    hidden_states_165 = self.L__mod___transformer_h_27_mlp_fc_out(hidden_states_164);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:285, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_27 = self.L__mod___transformer_h_27_mlp_dropout(hidden_states_165);  hidden_states_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_166 = attn_output_166 + feed_forward_hidden_states_27;  attn_output_166 = feed_forward_hidden_states_27 = None
    hidden_states_167 = add_166 + residual_27;  add_166 = residual_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    l__mod___transformer_ln_f = self.L__mod___transformer_ln_f(hidden_states_167);  hidden_states_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:715, code: hidden_states = hidden_states.view(output_shape)
    sequence_output = l__mod___transformer_ln_f.view((-1, 128, 4096));  l__mod___transformer_ln_f = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1122, code: logits = self.qa_outputs(sequence_output)
    logits = self.L__mod___qa_outputs(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1123, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_28 = logits.split(1, dim = -1);  logits = None
    start_logits = split_28[0]
    end_logits = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1124, code: start_logits = start_logits.squeeze(-1).contiguous()
    squeeze = start_logits.squeeze(-1);  start_logits = None
    start_logits_1 = squeeze.contiguous();  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1125, code: end_logits = end_logits.squeeze(-1).contiguous()
    squeeze_1 = end_logits.squeeze(-1);  end_logits = None
    end_logits_1 = squeeze_1.contiguous();  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1136, code: start_positions = start_positions.clamp(0, ignored_index)
    start_positions = l_cloned_inputs_start_positions_.clamp(0, 128);  l_cloned_inputs_start_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1137, code: end_positions = end_positions.clamp(0, ignored_index)
    end_positions = l_cloned_inputs_end_positions_.clamp(0, 128);  l_cloned_inputs_end_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    start_loss = torch.nn.functional.cross_entropy(start_logits_1, start_positions, None, None, 128, None, 'mean', 0.0);  start_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    end_loss = torch.nn.functional.cross_entropy(end_logits_1, end_positions, None, None, 128, None, 'mean', 0.0);  end_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1142, code: total_loss = (start_loss + end_loss) / 2
    add_168 = start_loss + end_loss;  start_loss = end_loss = None
    loss = add_168 / 2;  add_168 = None
    return (loss, start_logits_1, end_logits_1)
    