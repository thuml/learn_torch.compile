def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs['labels'], cloned_inputs[
        'decoder_input_ids'], cloned_inputs['input_ids'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3], graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7], graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11], graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15], graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19], graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23], graph_out_0[24],
        graph_out_0[25]), (graph_out_0[26], graph_out_0[27], graph_out_0[28],
        graph_out_0[29]), (graph_out_0[30], graph_out_0[31], graph_out_0[32],
        graph_out_0[33]), (graph_out_0[34], graph_out_0[35], graph_out_0[36],
        graph_out_0[37]), (graph_out_0[38], graph_out_0[39], graph_out_0[40],
        graph_out_0[41]), (graph_out_0[42], graph_out_0[43], graph_out_0[44],
        graph_out_0[45]), (graph_out_0[46], graph_out_0[47], graph_out_0[48],
        graph_out_0[49])), decoder_hidden_states=None, decoder_attentions=None,
        cross_attentions=None, encoder_last_hidden_state=graph_out_0[50],
        encoder_hidden_states=None, encoder_attentions=None)
    return __resume_at_100_4(graph_out_0[0].backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)
