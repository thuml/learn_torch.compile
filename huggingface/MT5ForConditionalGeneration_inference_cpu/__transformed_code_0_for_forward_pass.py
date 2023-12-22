def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['labels'], inputs
        ['decoder_input_ids'])
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
    tmp_72 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_73 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_74 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_75 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_76 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_77 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_78 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_79 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_80 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_81 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_82 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_83 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_84 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_85 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_86 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_87 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_88 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_89 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_90 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_91 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_92 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_93 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_94 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_95 = __import_torch_dot__dynamo_dot_utils.make_cell()
    import importlib
    tmp_95.cell_contents = mod.decoder.block[7].layer[1].EncDecAttention
    tmp_94.cell_contents = 1
    tmp_93.cell_contents = mod.decoder.block[7].layer[0].SelfAttention
    tmp_92.cell_contents = 1
    tmp_91.cell_contents = mod.decoder.block[6].layer[1].EncDecAttention
    tmp_90.cell_contents = 1
    tmp_89.cell_contents = mod.decoder.block[6].layer[0].SelfAttention
    tmp_88.cell_contents = 1
    tmp_87.cell_contents = mod.decoder.block[5].layer[1].EncDecAttention
    tmp_86.cell_contents = 1
    tmp_85.cell_contents = mod.decoder.block[5].layer[0].SelfAttention
    tmp_84.cell_contents = 1
    tmp_83.cell_contents = mod.decoder.block[4].layer[1].EncDecAttention
    tmp_82.cell_contents = 1
    tmp_81.cell_contents = mod.decoder.block[4].layer[0].SelfAttention
    tmp_80.cell_contents = 1
    tmp_79.cell_contents = mod.decoder.block[3].layer[1].EncDecAttention
    tmp_78.cell_contents = 1
    tmp_77.cell_contents = mod.decoder.block[3].layer[0].SelfAttention
    tmp_76.cell_contents = 1
    tmp_75.cell_contents = mod.decoder.block[2].layer[1].EncDecAttention
    tmp_74.cell_contents = 1
    tmp_73.cell_contents = mod.decoder.block[2].layer[0].SelfAttention
    tmp_72.cell_contents = 1
    tmp_71.cell_contents = mod.decoder.block[1].layer[1].EncDecAttention
    tmp_70.cell_contents = 1
    tmp_69.cell_contents = mod.decoder.block[1].layer[0].SelfAttention
    tmp_68.cell_contents = 1
    tmp_67.cell_contents = mod.decoder.block[0].layer[1].EncDecAttention
    tmp_66.cell_contents = 1
    tmp_65.cell_contents = mod.decoder.block[0].layer[0].SelfAttention
    tmp_64.cell_contents = 1
    tmp_63.cell_contents = mod.encoder.block[7].layer[0].SelfAttention
    tmp_62.cell_contents = 1
    tmp_61.cell_contents = mod.encoder.block[6].layer[0].SelfAttention
    tmp_60.cell_contents = 1
    tmp_59.cell_contents = mod.encoder.block[5].layer[0].SelfAttention
    tmp_58.cell_contents = 1
    tmp_57.cell_contents = mod.encoder.block[4].layer[0].SelfAttention
    tmp_56.cell_contents = 1
    tmp_55.cell_contents = mod.encoder.block[3].layer[0].SelfAttention
    tmp_54.cell_contents = 1
    tmp_53.cell_contents = mod.encoder.block[2].layer[0].SelfAttention
    tmp_52.cell_contents = 1
    tmp_51.cell_contents = mod.encoder.block[1].layer[0].SelfAttention
    tmp_50.cell_contents = 1
    tmp_49.cell_contents = mod.encoder.block[0].layer[0].SelfAttention
    tmp_48.cell_contents = 1
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3], graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7], graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11], graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15], graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19], graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23], graph_out_0[24],
        graph_out_0[25]), (graph_out_0[26], graph_out_0[27], graph_out_0[28],
        graph_out_0[29]), (graph_out_0[30], graph_out_0[31], graph_out_0[32],
        graph_out_0[33])), decoder_hidden_states=None, decoder_attentions=None,
        cross_attentions=None, encoder_last_hidden_state=graph_out_0[34],
        encoder_hidden_states=None, encoder_attentions=None)
