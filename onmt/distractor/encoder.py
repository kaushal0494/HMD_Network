"""Define encoder for distractor generation."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.misc import sequence_mask


class PermutationWrapper:
    """Sort the batch according to length, for using RNN pack/unpack"""
    def __init__(self, gpu, length, rnn_type='LSTM'):
        """
        :param inputs: [seq_length * batch_size]
        :param length: [each sequence length]
        :param batch_first: the first dimension of inputs denotes batch_size or not
        """

            
        device = torch.device("cuda" if gpu == True else "cpu")
        self.device = device
        self.original_length = length
        self.rnn_type = rnn_type
        # store original position in mapping,
        # e.g. mapping[1] = 5 denotes the tensor which currently in self.sorted_inputs position 5
        # originally locates in position 1 of self.original_inputs
        self.mapping = torch.zeros(self.original_length.size(0)).long().fill_(0)
 

    def sort(self, inputs, batch_first=False): 
       
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length
        """
        if batch_first:
            inputs = torch.transpose(inputs, 0, 1)  

        inputs_list = list(inputs_i.squeeze(1) for inputs_i in torch.split(inputs, 1, dim=1))
        
        sorted_inputs = sorted([(length_i.item(), i, inputs_i) for i, (length_i, inputs_i) in
                                enumerate(zip(self.original_length, inputs_list))], reverse=True)

        rnn_inputs = []
        rnn_length = []  
        for i, (length_i, original_idx, inputs_i) in enumerate(sorted_inputs):
            # original_idx: original position in the inputs
            self.mapping[original_idx] = i
            rnn_inputs.append(inputs_i)
            rnn_length.append(length_i)
        rnn_inputs = torch.stack(rnn_inputs, dim=1)
        rnn_length = torch.Tensor(rnn_length).type_as(self.original_length)
        return rnn_inputs, rnn_length


    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        """
        if self.rnn_type=='LSTM':
            remap_state = tuple(torch.index_select(state_i, 1, self.mapping.to(self.device))
                                for state_i in state)
        else:
            remap_state = torch.index_select(state, 1, self.mapping.to(self.device))

        remap_output = torch.index_select(output, 1, self.mapping.to(self.device))
        return remap_output, remap_state


class PermutationWrapper2D:
    """Permutation Wrapper for 2 levels input like sentence level/word level"""
    def __init__(self, gpu, word_length, sentence_length,
                 batch_first=False, rnn_type='LSTM'):
        """
        :param inputs: 3D input, [batch_size, sentence_seq_length, word_seq_length]
        :param word_length: number of tokens in each sentence
        :param sentence_length: number of sentences in each sample
        :param batch_first: batch_size in first dim of inputs or not
        :param rnn_type: LSTM/GRU
        """

        if batch_first:
            batch_size = word_length.size(0)
        else:
            batch_size = word_length.size(-1)

        device = torch.device("cuda" if gpu == True else "cpu")
        self.device = device
        self.batch_first = batch_first
        self.original_word_length = word_length
        self.original_sentence_length = sentence_length
        self.rnn_type = rnn_type
        self.sorted_inputs = []
        self.sorted_length = []
        # store original position in mapping,
        # e.g. mapping[1][3] = 5 denotes the tensor which currently
        # in self.sorted_inputs position 5
        # originally locates in position [1][3] of self.original_inputs
        self.mapping = torch.zeros(batch_size,
                                   sentence_length.max().item()).long().fill_(0)  # (batch_n,sent_n)

    def sort(self, inputs):
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length, effective_batch_size: true number of batches
        """
        # first reshape the src into a nested list, remove padded sentences
        inputs_list = list(inputs_i.squeeze(0) for inputs_i in torch.split(inputs, 1, 0))
        
        inputs_nested_list = []
        for sent_len_i, sent_i in zip(self.original_sentence_length, inputs_list):
            sent_tmp = list(words_i.squeeze(0) for words_i in torch.split(sent_i, 1, 0))     
            inputs_nested_list.append(sent_tmp[:sent_len_i])
        
        # remove 0 in word_length
        inputs_length_nested_list = []
        for sent_len_i, word_len_i in zip(self.original_sentence_length,
                                          self.original_word_length):
            inputs_length_nested_list.append(word_len_i[:sent_len_i])
        # get a orderedlist, each element: (word_len, sent_idx, word_ijdx, words)
        # sent_idx: i_th example in the batch
        # word_ijdx: j_th sentence sequence in the i_th example
        sorted_inputs = sorted([(sent_len_i[ij].item(), i, ij, word_ij)
                                for i, (sent_i, sent_len_i) in
                                enumerate(zip(inputs_nested_list, inputs_length_nested_list))
                                for ij, word_ij in enumerate(sent_i)], reverse=True)
        # sorted output
        rnn_inputs = []
        rnn_length = []
        for i, word_ij in enumerate(sorted_inputs):
            len_ij, ex_i, sent_ij, words_ij = word_ij
            self.mapping[ex_i, sent_ij] = i + 1  # i+1 because 0 is for empty.
            rnn_inputs.append(words_ij)
            rnn_length.append(len_ij)
        effective_batch_size = len(rnn_inputs)
        rnn_inputs = torch.stack(rnn_inputs, dim=1)
        rnn_length = torch.Tensor(rnn_length).type_as(self.original_word_length)
        
        return rnn_inputs, rnn_length, effective_batch_size

    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        here the returned batch is original_batch * max_len_sent, we need to reshape it further
        """
        # add a all_zero example at the first place
        output_padded = F.pad(output, (0, 0, 1, 0))
        remap_output = torch.index_select(output_padded, 1, self.mapping.view(-1).to(self.device))
        remap_output = remap_output.view(remap_output.size(0),
                                         self.mapping.size(0), self.mapping.size(1), -1)
        
        if self.rnn_type == "LSTM":
            h, c = state[0], state[1]
            h_padded = F.pad(h, (0, 0, 1, 0))
            c_padded = F.pad(c, (0, 0, 1, 0))
            remap_h = torch.index_select(h_padded, 1, self.mapping.view(-1).to(self.device))
            remap_c = torch.index_select(c_padded, 1, self.mapping.view(-1).to(self.device))
            remap_state = (remap_h.view(remap_h.size(0), self.mapping.size(0), self.mapping.size(1), -1),
                           remap_c.view(remap_c.size(0), self.mapping.size(0), self.mapping.size(1), -1))
        else:
            state_padded = F.pad(state, (0, 0, 1, 0))
            remap_state = torch.index_select(state_padded, 1, self.mapping.view(-1).to(self.device))
            remap_state = remap_state.view(remap_state.size(0), self.mapping.size(0), self.mapping.size(1), -1)    
        return remap_output, remap_state


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, emb_size=300):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, src_emb, lengths=None, side='tgt'):
        "See :obj:`EncoderBase.forward()`"
        packed_emb = src_emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            if side == 'tgt':
                packed_emb = pack(src_emb, lengths_list, enforce_sorted=False)
            else:  
                packed_emb = pack(src_emb, lengths_list)
                
        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        # here encoder_final[0][2,:,:] is the upper layer forward representation
        # here encoder_final[0][3,:,:] is the upper layer backward representation
        return memory_bank, encoder_final, lengths


class DistractorEncoder(nn.Module):
    """
    Distractor Generation Encoder
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, gpu, rnn_type,
                 word_encoder_type,  sent_encoder_type, question_init_type,
                 word_encoder_layers, sent_encoder_layers, question_init_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 l_ques=0.0, l_ans=0.0, tgt_embeddings=None):
        
        super(DistractorEncoder, self).__init__()
        assert embeddings is not None
        self.gpu = gpu
        self.device = torch.device("cuda" if gpu == True else "cpu")
        self.rnn_type = rnn_type
        self.embeddings = embeddings
        self.tgt_embeddings = tgt_embeddings
        self.l_ques = l_ques
        self.l_ans = l_ans

        # word encoder
        if word_encoder_type in ['brnn', 'rnn']:
            word_bidirectional = True if word_encoder_type=='brnn' else False
            word_dropout =  0.0 if word_encoder_layers == 1 else dropout
            self.word_encoder = RNNEncoder(rnn_type, word_bidirectional,
                                           word_encoder_layers, hidden_size,
                                           word_dropout, self.embeddings.embedding_size)
        else:
            raise NotImplementedError

        # sent encoder
        if sent_encoder_type in ['brnn', 'rnn']:
            sent_bidirectional = True if sent_encoder_type == 'brnn' else False
            sent_dropout = 0.0 if sent_encoder_layers == 1 else dropout
            self.sent_encoder = RNNEncoder(rnn_type, sent_bidirectional,
                                           sent_encoder_layers, hidden_size,
                                           sent_dropout, hidden_size)
        else:
            raise NotImplementedError

        # decoder hidden state initialization
        # here only use a unidirectional rnn to encode question
        if question_init_type in ['brnn', 'rnn']:
            init_bidirectional = True if question_init_type == 'brnn' else False
            ques_dropout = 0.0 if question_init_layers == 1 else dropout
            self.init_encoder = RNNEncoder(rnn_type, init_bidirectional,
                                           question_init_layers, hidden_size,
                                           ques_dropout, self.embeddings.embedding_size)
        else:
            raise NotImplementedError
            
            
        # decoder hidden state initialization
        # here only use a unidirectional rnn to encode gold distractor
        if question_init_type in ['brnn', 'rnn']:
            init_bidirectional = True if question_init_type == 'brnn' else False
            ques_dropout = 0.0 if question_init_layers == 1 else dropout
            self.tgt_encoder = RNNEncoder(rnn_type, init_bidirectional,
                                           question_init_layers, hidden_size,
                                           ques_dropout, self.tgt_embeddings.embedding_size)
        else:
            raise NotImplementedError    

        # static attention
        self.match_linear = nn.Linear(hidden_size, hidden_size)
        self.norm_linear = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_row = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        #projection Layer
        self.proj_linear = nn.Linear(self.embeddings.embedding_size, hidden_size)
        self.score_mult = nn.Linear(hidden_size, hidden_size)
        self.embd_proj_linear = nn.Linear(hidden_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.ss_ap_liner = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
    def gated_mechanism_of_averagepool(self, bank_sent, softsel_sent):
        ss_ap_z = self.sigmoid(self.ss_ap_liner(bank_sent) + self.ss_ap_liner(softsel_sent))
        return bank_sent * ss_ap_z +  softsel_sent * (1-ss_ap_z)
         
    def gated_mechanism(self, embd, hidden):
        embd_word_len, embd_batch, embd_dim = embd.size()
        hidden_word_len, hidden_batch, hidden_dim = hidden.size()
        assert embd_word_len == hidden_word_len 
        assert embd_batch == hidden_batch

        projection_embd = self.relu(self.proj_linear(embd.view(-1, embd_dim)))
        z = self.sigmoid(self.embd_proj_linear(projection_embd) + self.hidden_linear(hidden.view(-1, hidden_dim)))
        final_hidden = projection_embd * z +  hidden.view(-1, hidden_dim) * (1-z)
        return final_hidden.view(hidden_word_len, hidden_batch, hidden_dim)          
    
    def soft_sel2D_score(self, h_sent, h_passg):
        word_max_len, word_batch, sent_max_len, word_dim = h_passg.size()
        sent_max_len, sent_batch, sent_dim = h_sent.size()
        assert word_batch == sent_batch
        assert word_dim == sent_dim
        
        wh_passg = self.match_linear(h_passg.view(-1, word_dim)).view(word_batch, word_dim, -1)
        g = torch.bmm(h_sent.contiguous().transpose(0,1), wh_passg)
        g_bar = self.softmax(g)
        return torch.bmm(g_bar, h_passg.view(word_batch, -1, word_dim)).contiguous().transpose(0, 1) 
    
    def soft_sel1D_score(self, h_sent, h_passg):
        word_max_len, word_batch, word_dim = h_passg.size()
        sent_max_len, sent_batch, sent_dim = h_sent.size()
        assert word_batch == sent_batch
        assert word_dim == sent_dim
        
        wh_passg = self.match_linear(h_passg.view(-1, word_dim)).view(word_batch, word_dim, -1)
        g = torch.bmm(h_sent.contiguous().transpose(0,1), wh_passg)
        g_bar = self.softmax(g)
        return torch.bmm(g_bar, h_passg.transpose(0,1)).contiguous().transpose(0, 1)
   
    
    def dist_score(self, h_sent, h_passg):
        passg_word_len, passg_batch, passg_sent_len, passg_hid_dim = h_passg.size()
        sent_word_len, sent_batch, sent_hid_dim = h_sent.size()
        assert passg_batch == sent_batch
        assert passg_hid_dim == sent_hid_dim
        
        h_passg = h_passg.contiguous().transpose(0, 1).contiguous().transpose(1, 2)
        h_sent = h_sent.transpose(0, 1)
        h_sent_ = self.score_mult(h_sent.view(-1, sent_hid_dim))
        h_sent = h_sent_.view(sent_batch, sent_hid_dim, -1)
        
        soft_sel_score = torch.tensor([ [torch.sum(item) for item in torch.split(torch.bmm(sent.squeeze(1),  h_sent), 1, dim=0 ) ] for sent in torch.split(h_passg, 1, dim=1) ], requires_grad=True).to(self.device) 
     
        soft_sel_score_ = self.softmax(soft_sel_score.T)
        return soft_sel_score_   

    def score(self, h_ansques, h_passg):
        passg_batch, passg_len, passg_dim = h_passg.size()
        ansques_batch, ansques_dim = h_ansques.size()

        h_ansques_ = self.match_linear(h_ansques)
        h_ansques = h_ansques_.view(ansques_batch, 1, ansques_dim)
        h_passg_ = h_passg.transpose(1, 2)
        return torch.bmm(h_ansques, h_passg_)
    
    
    # Sorting, preparing Input for LSTM, Gated Machanism and Resorting
    def _feature_pw_fun2D(self, side_elems, word_length, sent_length):
        wrapped_inst = PermutationWrapper2D(self.gpu, word_length, sent_length, batch_first=True, \
                                                rnn_type=self.rnn_type)
        sorted_inst_list = [wrapped_inst.sort(side_elem)[0] for side_elem in side_elems]
        _, sorted_inst_length, _ = wrapped_inst.sort(side_elems[0]) 
        sorted_word = torch.cat([word.unsqueeze(-1) for word in sorted_inst_list], -1)  #cat words and feature  
        sorted_word_emb = self.embeddings(sorted_word)                                  #get embeddings
        sorted_word_bank, sorted_word_state, _ = self.word_encoder(sorted_word_emb, sorted_inst_length, 'src')
        word_bank, word_state = wrapped_inst.remap(sorted_word_bank, sorted_word_state)
        return word_bank, word_state        
    
    def _feature_pw_fun1D(self, side_elems, length, que_init):
        wrapped_inst = PermutationWrapper(self.gpu, length, rnn_type=self.rnn_type)
        sorted_inst_list = [wrapped_inst.sort(side_elem)[0] for side_elem in side_elems]
        _, sorted_inst_length = wrapped_inst.sort(side_elems[0]) 
        sorted_word = torch.cat([word.unsqueeze(-1) for word in sorted_inst_list], -1)     #cat words and feature
        sorted_word_emb = self.embeddings(sorted_word)                                     #get embeddings                

        if que_init:
            sorted_word_bank, sorted_word_state, _ = self.init_encoder(sorted_word_emb, sorted_inst_length, 'ques')
            word_bank, word_state = wrapped_inst.remap(sorted_word_bank, sorted_word_state)
            return word_bank, word_state
        else:
            sorted_word_bank, sorted_word_state, _ = self.word_encoder(sorted_word_emb, sorted_inst_length, 'ans')
            word_bank, word_state = wrapped_inst.remap(sorted_word_bank, sorted_word_state)
            return word_bank, word_state                                                                     
        
        
        
    def forward(self, src, ques, ans, sent_length, word_length, ques_length, ans_length, tgt, tgt_length):        
        # word bank dim: words X batch X sentences X hidden_dim
        #word_state: tuple of size 2(encoder LSTM): (direction*layers) X batch X hidden 
        word_bank, word_state = self._feature_pw_fun2D(src, word_length, sent_length)      
         
        #Same for question init, question, sentence answer
        # bank dim: words X batch X hidden_dim
        # state dim: tuple of size 2(encoder LSTM): (direction*layers) X batch X hidden
        quesinit_bank, quesinit_state = self._feature_pw_fun1D(ques, ques_length, True)  
        ans_bank, ans_state = self._feature_pw_fun1D(ans, ans_length, False)
        ques_bank, ques_state = self._feature_pw_fun1D(ques, ques_length, False)        

        ## sentence level
        _, bs, sentlen, hid = word_state[0].size()
        # [2n, bs, sentlen, hid] -> [sentlen, bs, 2n, hid] -> [sentlen, bs, hid * 2n]
        sent_emb = word_state[0].transpose(0, 2)[:, :, -2:, :].contiguous().view(sentlen, bs, -1)
        wrapped_sent = PermutationWrapper(self.gpu, sent_length, rnn_type=self.rnn_type)
        sorted_sent_emb, sorted_sent_length = wrapped_sent.sort(sent_emb)
        sorted_sent_bank, sorted_sent_state, _ = self.sent_encoder(sorted_sent_emb, sorted_sent_length, 'sent')        
        sent_bank, sent_state = wrapped_sent.remap(sorted_sent_bank, sorted_sent_state)
        
        #tgt
        tgt_word_emb = self.tgt_embeddings(tgt.unsqueeze(-1))
        tgt_word_bank, tgt_state, _ = self.tgt_encoder(tgt_word_emb, tgt_length, "tgt")
        
        #softsel
        H_ques = self.soft_sel2D_score(ques_bank, word_bank)
        H_ans = self.soft_sel2D_score(ans_bank, word_bank)

        H_ques_ans = self.soft_sel1D_score(ques_bank, ans_bank)   #add
        H_ans_ques = self.soft_sel1D_score(ans_bank, ques_bank)   #add
        H_ques_bar = self.soft_sel2D_score(H_ques_ans, word_bank) #add
        H_ans_bar = self.soft_sel2D_score(H_ans_ques, word_bank)  #add
     
        #Average pooling of word_bank, question_bank and ans_bank 
        match_word = torch.div(word_bank.sum(0), (word_length.unsqueeze(-1).float() + 1e-20))   #passage Sentence
        match_ques = torch.div(ques_bank.sum(0), (ques_length.unsqueeze(-1).float() + 1e-20))   #ques 
        match_ans = torch.div(ans_bank.sum(0), ans_length.unsqueeze(-1).float() + 1e-20)        #ans
        
        #Average pooling of softsel question and softsel answer 
        H_match_ques = torch.div(ques_bank.sum(0), (ques_length.unsqueeze(-1).float() + 1e-20))   #H_ques
        H_match_ans = torch.div(H_ans.sum(0), ans_length.unsqueeze(-1).float() + 1e-20)           #H_ans

        H_match_ques_bar = torch.div(H_ques_bar.sum(0), ques_length.unsqueeze(-1).float() + 1e-20)           #H_ques_bar #add
        H_match_ans_bar = torch.div(H_ans_bar.sum(0), ans_length.unsqueeze(-1).float() + 1e-20)           #H_ans_bar #add
         
        #gated_question and gated answer
        gated_ques = self.gated_mechanism_of_averagepool(match_ques, H_match_ques)
        gated_ans = self.gated_mechanism_of_averagepool(match_ans, H_match_ans)

        gated_ques_bar = self.gated_mechanism_of_averagepool(match_ans, H_match_ques_bar)   #add
        gated_ans_bar = self.gated_mechanism_of_averagepool(match_ans, H_match_ans_bar)     #add
        
        match_score = (self.l_ques * self.score(gated_ques, match_word) - (self.l_ans/3) * (self.score(gated_ans, match_word) + self.score(gated_ques_bar, match_word) +self.score(gated_ans_bar, match_word) )   ).squeeze(1)      #add
        # calculating temprature 
        temperature = self.sigmoid(self.norm_linear(match_ques)) + 1e-20
        #final static attention: batch X sentences 
        static_attn = torch.div(match_score, temperature) + 1e-20
        #static_attn = self.softmax(static_attn)   
        return word_bank, sent_bank, quesinit_state, static_attn, tgt_state
