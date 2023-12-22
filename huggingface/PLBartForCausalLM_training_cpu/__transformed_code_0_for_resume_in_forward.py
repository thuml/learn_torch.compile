def __transformed_code_0_for_resume_in_forward(___stack0, self, labels, return_dict):
    attention_mask = None; cross_attn_head_mask = None; encoder_attention_mask = None; encoder_hidden_states = None; head_mask = None; input_ids = None; inputs_embeds = None; logits = None; loss = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; outputs = None; past_key_values = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_20(___stack0.last_hidden_state, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithCrossAttentions(loss=graph_out_0[0], logits=
        graph_out_0[1], past_key_values=___stack0.past_key_values,
        hidden_states=___stack0.hidden_states, attentions=___stack0.attentions,
        cross_attentions=___stack0.cross_attentions)
