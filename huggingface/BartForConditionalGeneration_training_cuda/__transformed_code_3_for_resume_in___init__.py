def __transformed_code_3_for_resume_in___init__(self, cross_attentions, encoder_last_hidden_state, encoder_hidden_states, encoder_attentions):
    decoder_attentions = None; decoder_hidden_states = None; last_hidden_state = None; past_key_values = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    self.cross_attentions = cross_attentions
    return __resume_at_32_46(self, encoder_last_hidden_state,
        encoder_hidden_states, encoder_attentions)
