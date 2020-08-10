""" memory network Model base class definition """
import torch.nn as nn
import torch

class DGModel(nn.Module):
    """
    Core trainable object in Distractor Generation. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(DGModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, src, ques, ans, tgt, sent_lengths, word_lengths, ques_length, ans_length, tgt_length):
        tgt = tgt[0]
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_length = tgt_length -1

        word_mem_bank, sent_mem_bank, quesinit, static_attn, tgt_state = self.encoder(
            src, ques, ans, sent_lengths, word_lengths, ques_length, ans_length, tgt, tgt_length)

        enc_state1 = self.decoder.init_decoder_state(quesinit)
        enc_state2 = self.decoder.init_decoder_state(quesinit)
        enc_state3 = self.decoder.init_decoder_state(quesinit)
        
        # update inputfeed by using question last hidden state
        enc_state1.update_state(enc_state1.hidden, enc_state1.hidden[0][-1].unsqueeze(0), enc_state1.coverage)
        enc_state2.update_state(enc_state2.hidden, enc_state2.hidden[0][-1].unsqueeze(0), enc_state2.coverage)
        enc_state3.update_state(enc_state3.hidden, enc_state3.hidden[0][-1].unsqueeze(0), enc_state3.coverage)
        
        dec_h1, decoder_outputs1, dec_state1, attns1, dec_h2, decoder_outputs2, dec_state2, attns2, dec_h3,\
        decoder_outputs3, dec_state3, attns3 = self.decoder(tgt, word_mem_bank, sent_mem_bank, enc_state1 , enc_state2, \
                                                            enc_state3, word_lengths, sent_lengths, static_attn)
        
        #calculating Cosine Similarity Score
        tgt_hidden = tgt_state[0].squeeze(0)
        dec_hidden1 = dec_h1[0].squeeze(0)
        dec_hidden2 = dec_h2[0].squeeze(0)
        dec_hidden3 = dec_h3[0].squeeze(0)
        
        dis1_sim = torch.sum(self.cos(tgt_hidden, dec_hidden1))
        dis2_sim = torch.sum(self.cos(tgt_hidden, dec_hidden2))
        dis3_sim = torch.sum(self.cos(tgt_hidden, dec_hidden3))
        
        dis12_sim = torch.sum(self.cos(dec_hidden1, dec_hidden2))
        dis13_sim = torch.sum(self.cos(dec_hidden1, dec_hidden3))
        dis23_sim = torch.sum(self.cos(dec_hidden2, dec_hidden3))
                
        return decoder_outputs1, attns1, dec_state1, decoder_outputs2, attns2, dec_state2, decoder_outputs3, attns3, \
    dec_state3, dis1_sim, dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim  
