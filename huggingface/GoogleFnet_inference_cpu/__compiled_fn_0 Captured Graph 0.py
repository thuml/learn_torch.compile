from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:586, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    l__mod___fnet_embeddings_token_type_ids = self.L__mod___fnet_embeddings_token_type_ids
    buffered_token_type_ids = l__mod___fnet_embeddings_token_type_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___fnet_embeddings_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:587, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids.expand(1, 512);  buffered_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:134, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___fnet_embeddings_position_ids = self.L__mod___fnet_embeddings_position_ids
    position_ids = l__mod___fnet_embeddings_position_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___fnet_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:148, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__mod___fnet_embeddings_word_embeddings(l_inputs_input_ids_);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:149, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__mod___fnet_embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:151, code: embeddings = inputs_embeds + token_type_embeddings
    embeddings = inputs_embeds + token_type_embeddings;  inputs_embeds = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:153, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__mod___fnet_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:154, code: embeddings += position_embeddings
    embeddings += position_embeddings;  embeddings_1 = embeddings;  embeddings = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    embeddings_2 = self.L__mod___fnet_embeddings_LayerNorm(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    embeddings_3 = self.L__mod___fnet_embeddings_projection(embeddings_2);  embeddings_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:157, code: embeddings = self.dropout(embeddings)
    embedding_output = self.L__mod___fnet_embeddings_dropout(embeddings_3);  embeddings_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn = torch._C._fft.fft_fftn(embedding_output, dim = (1, 2))
    outputs = fft_fftn.real;  fft_fftn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_1 = embedding_output + outputs;  embedding_output = outputs = None
    fourier_output = self.L__mod___fnet_encoder_layer_0_fourier_output_LayerNorm(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_1 = self.L__mod___fnet_encoder_layer_0_intermediate_dense(fourier_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul = 0.5 * hidden_states_1
    pow_1 = torch.pow(hidden_states_1, 3.0)
    mul_1 = 0.044715 * pow_1;  pow_1 = None
    add_2 = hidden_states_1 + mul_1;  hidden_states_1 = mul_1 = None
    mul_2 = 0.7978845608028654 * add_2;  add_2 = None
    tanh = torch.tanh(mul_2);  mul_2 = None
    add_3 = 1.0 + tanh;  tanh = None
    intermediate_output = mul * add_3;  mul = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_3 = self.L__mod___fnet_encoder_layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_4 = self.L__mod___fnet_encoder_layer_0_output_dropout(hidden_states_3);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4 = hidden_states_4 + fourier_output;  hidden_states_4 = fourier_output = None
    hidden_states_6 = self.L__mod___fnet_encoder_layer_0_output_LayerNorm(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_1 = torch._C._fft.fft_fftn(hidden_states_6, dim = (1, 2))
    outputs_1 = fft_fftn_1.real;  fft_fftn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_5 = hidden_states_6 + outputs_1;  hidden_states_6 = outputs_1 = None
    fourier_output_2 = self.L__mod___fnet_encoder_layer_1_fourier_output_LayerNorm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_8 = self.L__mod___fnet_encoder_layer_1_intermediate_dense(fourier_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4 = 0.5 * hidden_states_8
    pow_2 = torch.pow(hidden_states_8, 3.0)
    mul_5 = 0.044715 * pow_2;  pow_2 = None
    add_6 = hidden_states_8 + mul_5;  hidden_states_8 = mul_5 = None
    mul_6 = 0.7978845608028654 * add_6;  add_6 = None
    tanh_1 = torch.tanh(mul_6);  mul_6 = None
    add_7 = 1.0 + tanh_1;  tanh_1 = None
    intermediate_output_1 = mul_4 * add_7;  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_10 = self.L__mod___fnet_encoder_layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_11 = self.L__mod___fnet_encoder_layer_1_output_dropout(hidden_states_10);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8 = hidden_states_11 + fourier_output_2;  hidden_states_11 = fourier_output_2 = None
    hidden_states_13 = self.L__mod___fnet_encoder_layer_1_output_LayerNorm(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_2 = torch._C._fft.fft_fftn(hidden_states_13, dim = (1, 2))
    outputs_2 = fft_fftn_2.real;  fft_fftn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_9 = hidden_states_13 + outputs_2;  hidden_states_13 = outputs_2 = None
    fourier_output_4 = self.L__mod___fnet_encoder_layer_2_fourier_output_LayerNorm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_15 = self.L__mod___fnet_encoder_layer_2_intermediate_dense(fourier_output_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_8 = 0.5 * hidden_states_15
    pow_3 = torch.pow(hidden_states_15, 3.0)
    mul_9 = 0.044715 * pow_3;  pow_3 = None
    add_10 = hidden_states_15 + mul_9;  hidden_states_15 = mul_9 = None
    mul_10 = 0.7978845608028654 * add_10;  add_10 = None
    tanh_2 = torch.tanh(mul_10);  mul_10 = None
    add_11 = 1.0 + tanh_2;  tanh_2 = None
    intermediate_output_2 = mul_8 * add_11;  mul_8 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_17 = self.L__mod___fnet_encoder_layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_18 = self.L__mod___fnet_encoder_layer_2_output_dropout(hidden_states_17);  hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_12 = hidden_states_18 + fourier_output_4;  hidden_states_18 = fourier_output_4 = None
    hidden_states_20 = self.L__mod___fnet_encoder_layer_2_output_LayerNorm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_3 = torch._C._fft.fft_fftn(hidden_states_20, dim = (1, 2))
    outputs_3 = fft_fftn_3.real;  fft_fftn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_13 = hidden_states_20 + outputs_3;  hidden_states_20 = outputs_3 = None
    fourier_output_6 = self.L__mod___fnet_encoder_layer_3_fourier_output_LayerNorm(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_22 = self.L__mod___fnet_encoder_layer_3_intermediate_dense(fourier_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12 = 0.5 * hidden_states_22
    pow_4 = torch.pow(hidden_states_22, 3.0)
    mul_13 = 0.044715 * pow_4;  pow_4 = None
    add_14 = hidden_states_22 + mul_13;  hidden_states_22 = mul_13 = None
    mul_14 = 0.7978845608028654 * add_14;  add_14 = None
    tanh_3 = torch.tanh(mul_14);  mul_14 = None
    add_15 = 1.0 + tanh_3;  tanh_3 = None
    intermediate_output_3 = mul_12 * add_15;  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__mod___fnet_encoder_layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_25 = self.L__mod___fnet_encoder_layer_3_output_dropout(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_16 = hidden_states_25 + fourier_output_6;  hidden_states_25 = fourier_output_6 = None
    hidden_states_27 = self.L__mod___fnet_encoder_layer_3_output_LayerNorm(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_4 = torch._C._fft.fft_fftn(hidden_states_27, dim = (1, 2))
    outputs_4 = fft_fftn_4.real;  fft_fftn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_17 = hidden_states_27 + outputs_4;  hidden_states_27 = outputs_4 = None
    fourier_output_8 = self.L__mod___fnet_encoder_layer_4_fourier_output_LayerNorm(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_29 = self.L__mod___fnet_encoder_layer_4_intermediate_dense(fourier_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16 = 0.5 * hidden_states_29
    pow_5 = torch.pow(hidden_states_29, 3.0)
    mul_17 = 0.044715 * pow_5;  pow_5 = None
    add_18 = hidden_states_29 + mul_17;  hidden_states_29 = mul_17 = None
    mul_18 = 0.7978845608028654 * add_18;  add_18 = None
    tanh_4 = torch.tanh(mul_18);  mul_18 = None
    add_19 = 1.0 + tanh_4;  tanh_4 = None
    intermediate_output_4 = mul_16 * add_19;  mul_16 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_31 = self.L__mod___fnet_encoder_layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_32 = self.L__mod___fnet_encoder_layer_4_output_dropout(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_20 = hidden_states_32 + fourier_output_8;  hidden_states_32 = fourier_output_8 = None
    hidden_states_34 = self.L__mod___fnet_encoder_layer_4_output_LayerNorm(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_5 = torch._C._fft.fft_fftn(hidden_states_34, dim = (1, 2))
    outputs_5 = fft_fftn_5.real;  fft_fftn_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_21 = hidden_states_34 + outputs_5;  hidden_states_34 = outputs_5 = None
    fourier_output_10 = self.L__mod___fnet_encoder_layer_5_fourier_output_LayerNorm(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_36 = self.L__mod___fnet_encoder_layer_5_intermediate_dense(fourier_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20 = 0.5 * hidden_states_36
    pow_6 = torch.pow(hidden_states_36, 3.0)
    mul_21 = 0.044715 * pow_6;  pow_6 = None
    add_22 = hidden_states_36 + mul_21;  hidden_states_36 = mul_21 = None
    mul_22 = 0.7978845608028654 * add_22;  add_22 = None
    tanh_5 = torch.tanh(mul_22);  mul_22 = None
    add_23 = 1.0 + tanh_5;  tanh_5 = None
    intermediate_output_5 = mul_20 * add_23;  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_38 = self.L__mod___fnet_encoder_layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_39 = self.L__mod___fnet_encoder_layer_5_output_dropout(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24 = hidden_states_39 + fourier_output_10;  hidden_states_39 = fourier_output_10 = None
    hidden_states_41 = self.L__mod___fnet_encoder_layer_5_output_LayerNorm(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_6 = torch._C._fft.fft_fftn(hidden_states_41, dim = (1, 2))
    outputs_6 = fft_fftn_6.real;  fft_fftn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_25 = hidden_states_41 + outputs_6;  hidden_states_41 = outputs_6 = None
    fourier_output_12 = self.L__mod___fnet_encoder_layer_6_fourier_output_LayerNorm(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_43 = self.L__mod___fnet_encoder_layer_6_intermediate_dense(fourier_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_24 = 0.5 * hidden_states_43
    pow_7 = torch.pow(hidden_states_43, 3.0)
    mul_25 = 0.044715 * pow_7;  pow_7 = None
    add_26 = hidden_states_43 + mul_25;  hidden_states_43 = mul_25 = None
    mul_26 = 0.7978845608028654 * add_26;  add_26 = None
    tanh_6 = torch.tanh(mul_26);  mul_26 = None
    add_27 = 1.0 + tanh_6;  tanh_6 = None
    intermediate_output_6 = mul_24 * add_27;  mul_24 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_45 = self.L__mod___fnet_encoder_layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_46 = self.L__mod___fnet_encoder_layer_6_output_dropout(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28 = hidden_states_46 + fourier_output_12;  hidden_states_46 = fourier_output_12 = None
    hidden_states_48 = self.L__mod___fnet_encoder_layer_6_output_LayerNorm(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_7 = torch._C._fft.fft_fftn(hidden_states_48, dim = (1, 2))
    outputs_7 = fft_fftn_7.real;  fft_fftn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_29 = hidden_states_48 + outputs_7;  hidden_states_48 = outputs_7 = None
    fourier_output_14 = self.L__mod___fnet_encoder_layer_7_fourier_output_LayerNorm(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_50 = self.L__mod___fnet_encoder_layer_7_intermediate_dense(fourier_output_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28 = 0.5 * hidden_states_50
    pow_8 = torch.pow(hidden_states_50, 3.0)
    mul_29 = 0.044715 * pow_8;  pow_8 = None
    add_30 = hidden_states_50 + mul_29;  hidden_states_50 = mul_29 = None
    mul_30 = 0.7978845608028654 * add_30;  add_30 = None
    tanh_7 = torch.tanh(mul_30);  mul_30 = None
    add_31 = 1.0 + tanh_7;  tanh_7 = None
    intermediate_output_7 = mul_28 * add_31;  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_52 = self.L__mod___fnet_encoder_layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_53 = self.L__mod___fnet_encoder_layer_7_output_dropout(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32 = hidden_states_53 + fourier_output_14;  hidden_states_53 = fourier_output_14 = None
    hidden_states_55 = self.L__mod___fnet_encoder_layer_7_output_LayerNorm(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_8 = torch._C._fft.fft_fftn(hidden_states_55, dim = (1, 2))
    outputs_8 = fft_fftn_8.real;  fft_fftn_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_33 = hidden_states_55 + outputs_8;  hidden_states_55 = outputs_8 = None
    fourier_output_16 = self.L__mod___fnet_encoder_layer_8_fourier_output_LayerNorm(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_57 = self.L__mod___fnet_encoder_layer_8_intermediate_dense(fourier_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_32 = 0.5 * hidden_states_57
    pow_9 = torch.pow(hidden_states_57, 3.0)
    mul_33 = 0.044715 * pow_9;  pow_9 = None
    add_34 = hidden_states_57 + mul_33;  hidden_states_57 = mul_33 = None
    mul_34 = 0.7978845608028654 * add_34;  add_34 = None
    tanh_8 = torch.tanh(mul_34);  mul_34 = None
    add_35 = 1.0 + tanh_8;  tanh_8 = None
    intermediate_output_8 = mul_32 * add_35;  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_59 = self.L__mod___fnet_encoder_layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_60 = self.L__mod___fnet_encoder_layer_8_output_dropout(hidden_states_59);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36 = hidden_states_60 + fourier_output_16;  hidden_states_60 = fourier_output_16 = None
    hidden_states_62 = self.L__mod___fnet_encoder_layer_8_output_LayerNorm(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_9 = torch._C._fft.fft_fftn(hidden_states_62, dim = (1, 2))
    outputs_9 = fft_fftn_9.real;  fft_fftn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_37 = hidden_states_62 + outputs_9;  hidden_states_62 = outputs_9 = None
    fourier_output_18 = self.L__mod___fnet_encoder_layer_9_fourier_output_LayerNorm(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_64 = self.L__mod___fnet_encoder_layer_9_intermediate_dense(fourier_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36 = 0.5 * hidden_states_64
    pow_10 = torch.pow(hidden_states_64, 3.0)
    mul_37 = 0.044715 * pow_10;  pow_10 = None
    add_38 = hidden_states_64 + mul_37;  hidden_states_64 = mul_37 = None
    mul_38 = 0.7978845608028654 * add_38;  add_38 = None
    tanh_9 = torch.tanh(mul_38);  mul_38 = None
    add_39 = 1.0 + tanh_9;  tanh_9 = None
    intermediate_output_9 = mul_36 * add_39;  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_66 = self.L__mod___fnet_encoder_layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_67 = self.L__mod___fnet_encoder_layer_9_output_dropout(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_40 = hidden_states_67 + fourier_output_18;  hidden_states_67 = fourier_output_18 = None
    hidden_states_69 = self.L__mod___fnet_encoder_layer_9_output_LayerNorm(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_10 = torch._C._fft.fft_fftn(hidden_states_69, dim = (1, 2))
    outputs_10 = fft_fftn_10.real;  fft_fftn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_41 = hidden_states_69 + outputs_10;  hidden_states_69 = outputs_10 = None
    fourier_output_20 = self.L__mod___fnet_encoder_layer_10_fourier_output_LayerNorm(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_71 = self.L__mod___fnet_encoder_layer_10_intermediate_dense(fourier_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_40 = 0.5 * hidden_states_71
    pow_11 = torch.pow(hidden_states_71, 3.0)
    mul_41 = 0.044715 * pow_11;  pow_11 = None
    add_42 = hidden_states_71 + mul_41;  hidden_states_71 = mul_41 = None
    mul_42 = 0.7978845608028654 * add_42;  add_42 = None
    tanh_10 = torch.tanh(mul_42);  mul_42 = None
    add_43 = 1.0 + tanh_10;  tanh_10 = None
    intermediate_output_10 = mul_40 * add_43;  mul_40 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_73 = self.L__mod___fnet_encoder_layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_74 = self.L__mod___fnet_encoder_layer_10_output_dropout(hidden_states_73);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_44 = hidden_states_74 + fourier_output_20;  hidden_states_74 = fourier_output_20 = None
    hidden_states_76 = self.L__mod___fnet_encoder_layer_10_output_LayerNorm(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    fft_fftn_11 = torch._C._fft.fft_fftn(hidden_states_76, dim = (1, 2))
    outputs_11 = fft_fftn_11.real;  fft_fftn_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_45 = hidden_states_76 + outputs_11;  hidden_states_76 = outputs_11 = None
    fourier_output_22 = self.L__mod___fnet_encoder_layer_11_fourier_output_LayerNorm(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    hidden_states_78 = self.L__mod___fnet_encoder_layer_11_intermediate_dense(fourier_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44 = 0.5 * hidden_states_78
    pow_12 = torch.pow(hidden_states_78, 3.0)
    mul_45 = 0.044715 * pow_12;  pow_12 = None
    add_46 = hidden_states_78 + mul_45;  hidden_states_78 = mul_45 = None
    mul_46 = 0.7978845608028654 * add_46;  add_46 = None
    tanh_11 = torch.tanh(mul_46);  mul_46 = None
    add_47 = 1.0 + tanh_11;  tanh_11 = None
    intermediate_output_11 = mul_44 * add_47;  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    hidden_states_80 = self.L__mod___fnet_encoder_layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    hidden_states_81 = self.L__mod___fnet_encoder_layer_11_output_dropout(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_48 = hidden_states_81 + fourier_output_22;  hidden_states_81 = fourier_output_22 = None
    sequence_output = self.L__mod___fnet_encoder_layer_11_output_LayerNorm(add_48);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:327, code: first_token_tensor = hidden_states[:, 0]
    first_token_tensor = sequence_output[(slice(None, None, None), 0)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:328, code: pooled_output = self.dense(first_token_tensor)
    pooled_output = self.L__mod___fnet_pooler_dense(first_token_tensor);  first_token_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:329, code: pooled_output = self.activation(pooled_output)
    pooler_output = self.L__mod___fnet_pooler_activation(pooled_output);  pooled_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    hidden_states_84 = self.L__mod___cls_predictions_transform_dense(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_48 = 0.5 * hidden_states_84
    pow_13 = torch.pow(hidden_states_84, 3.0)
    mul_49 = 0.044715 * pow_13;  pow_13 = None
    add_49 = hidden_states_84 + mul_49;  hidden_states_84 = mul_49 = None
    mul_50 = 0.7978845608028654 * add_49;  add_49 = None
    tanh_12 = torch.tanh(mul_50);  mul_50 = None
    add_50 = 1.0 + tanh_12;  tanh_12 = None
    hidden_states_85 = mul_48 * add_50;  mul_48 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    hidden_states_87 = self.L__mod___cls_predictions_transform_LayerNorm(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    prediction_scores = self.L__mod___cls_predictions_decoder(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view = prediction_scores.view(-1, 32000)
    view_1 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (masked_lm_loss, prediction_scores)
    