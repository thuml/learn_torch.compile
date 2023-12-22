def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs[0], cloned_inputs[1])
    tmp_36 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_37 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_38 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_39 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_40 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_41 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_42 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_43 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_44 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_45 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_46 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_47 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_48 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_49 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_50 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_51 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_52 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_53 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_54 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_55 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_56 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_57 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_58 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_59 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_60 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_61 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_62 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_63 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_64 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_65 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_66 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_67 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_68 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_69 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_70 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_71 = __import_torch_dot__dynamo_dot_utils.make_cell()
    import importlib
    __temp_45 = importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=None, logits=graph_out_0[0], past_key_values=((
        graph_out_0[1], graph_out_0[2], graph_out_0[3], graph_out_0[4]), (
        graph_out_0[5], graph_out_0[6], graph_out_0[7], graph_out_0[8]), (
        graph_out_0[9], graph_out_0[10], graph_out_0[11], graph_out_0[12]), (
        graph_out_0[13], graph_out_0[14], graph_out_0[15], graph_out_0[16]), (
        graph_out_0[17], graph_out_0[18], graph_out_0[19], graph_out_0[20]), (
        graph_out_0[21], graph_out_0[22], graph_out_0[23], graph_out_0[24])),
        decoder_hidden_states=None, decoder_attentions=None, cross_attentions=
        None, encoder_last_hidden_state=graph_out_0[25], encoder_hidden_states=
        None, encoder_attentions=None)
    tmp_71.cell_contents = mod.model.decoder.block[5].layer[1].EncDecAttention
    tmp_70.cell_contents = 4
    tmp_69.cell_contents = mod.model.decoder.block[5].layer[0].SelfAttention
    tmp_68.cell_contents = 4
    tmp_67.cell_contents = mod.model.decoder.block[4].layer[1].EncDecAttention
    tmp_66.cell_contents = 4
    tmp_65.cell_contents = mod.model.decoder.block[4].layer[0].SelfAttention
    tmp_64.cell_contents = 4
    tmp_63.cell_contents = mod.model.decoder.block[3].layer[1].EncDecAttention
    tmp_62.cell_contents = 4
    tmp_61.cell_contents = mod.model.decoder.block[3].layer[0].SelfAttention
    tmp_60.cell_contents = 4
    tmp_59.cell_contents = mod.model.decoder.block[2].layer[1].EncDecAttention
    tmp_58.cell_contents = 4
    tmp_57.cell_contents = mod.model.decoder.block[2].layer[0].SelfAttention
    tmp_56.cell_contents = 4
    tmp_55.cell_contents = mod.model.decoder.block[1].layer[1].EncDecAttention
    tmp_54.cell_contents = 4
    tmp_53.cell_contents = mod.model.decoder.block[1].layer[0].SelfAttention
    tmp_52.cell_contents = 4
    tmp_51.cell_contents = mod.model.decoder.block[0].layer[1].EncDecAttention
    tmp_50.cell_contents = 4
    tmp_49.cell_contents = mod.model.decoder.block[0].layer[0].SelfAttention
    tmp_48.cell_contents = 4
    tmp_47.cell_contents = mod.model.encoder.block[5].layer[0].SelfAttention
    tmp_46.cell_contents = 4
    tmp_45.cell_contents = mod.model.encoder.block[4].layer[0].SelfAttention
    tmp_44.cell_contents = 4
    tmp_43.cell_contents = mod.model.encoder.block[3].layer[0].SelfAttention
    tmp_42.cell_contents = 4
    tmp_41.cell_contents = mod.model.encoder.block[2].layer[0].SelfAttention
    tmp_40.cell_contents = 4
    tmp_39.cell_contents = mod.model.encoder.block[1].layer[0].SelfAttention
    tmp_38.cell_contents = 4
    tmp_37.cell_contents = mod.model.encoder.block[0].layer[0].SelfAttention
    tmp_36.cell_contents = 4
    pred = __temp_45
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_48 = self.compute_loss(__temp_45)
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_48_5(__import_contextlib.nullcontext, __temp_48, self,
        mod, collect_outputs, cloned_inputs, pred)
