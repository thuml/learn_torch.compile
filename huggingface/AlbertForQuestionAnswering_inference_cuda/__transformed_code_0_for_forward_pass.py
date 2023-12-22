def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['start_positions'
        ], inputs['end_positions'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).QuestionAnsweringModelOutput(loss=graph_out_0[0], start_logits=
        graph_out_0[1], end_logits=graph_out_0[2], hidden_states=None,
        attentions=None)
