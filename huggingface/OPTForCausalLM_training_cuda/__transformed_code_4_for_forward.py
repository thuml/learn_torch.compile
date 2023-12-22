def __transformed_code_4_for_forward(self, hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions, use_cache):
    hidden_states_shape = None; outputs = None; present_key_value = None; residual = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_10(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])
