def __transformed_code_2_for_resume_in___init__(self, decoder_hidden_states, decoder_attentions, cross_attentions, encoder_last_hidden_state, encoder_hidden_states, encoder_attentions):
    last_hidden_state = None; past_key_values = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    self.decoder_hidden_states = decoder_hidden_states
    return __resume_at_20_35(self, decoder_attentions, cross_attentions,
        encoder_last_hidden_state, encoder_hidden_states, encoder_attentions)
