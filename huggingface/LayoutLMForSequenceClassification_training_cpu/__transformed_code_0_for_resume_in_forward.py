def __transformed_code_0_for_resume_in_forward(self, labels, return_dict, outputs, logits, loss):
    attention_mask = None; bbox = None; head_mask = None; input_ids = None; inputs_embeds = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; pooled_output = None; position_ids = None; token_type_ids = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_7(logits, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).SequenceClassifierOutput(loss=graph_out_0[0], logits=logits,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
