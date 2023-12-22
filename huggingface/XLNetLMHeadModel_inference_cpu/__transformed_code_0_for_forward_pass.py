def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['labels'])
    import importlib
    return importlib.import_module('transformers.models.xlnet.modeling_xlnet'
        ).XLNetLMHeadModelOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        mems=(graph_out_0[2], graph_out_0[3], graph_out_0[4], graph_out_0[5],
        graph_out_0[6], graph_out_0[7], graph_out_0[8], graph_out_0[9],
        graph_out_0[10], graph_out_0[11], graph_out_0[12], graph_out_0[13],
        graph_out_0[14], graph_out_0[15], graph_out_0[16], graph_out_0[17],
        graph_out_0[18], graph_out_0[19], graph_out_0[20], graph_out_0[21],
        graph_out_0[22], graph_out_0[23], graph_out_0[24], graph_out_0[25]),
        hidden_states=None, attentions=None)
