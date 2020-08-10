""" Hierarchical attention modules """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd


class HierarchicalAttention(nn.Module):
    """Dynamic attention"""
    def __init__(self, gpu, dim, attn_type="general"):
        super(HierarchicalAttention, self).__init__()
 
        device = torch.device("cuda" if gpu == True else "cpu")
        self.device = device
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        # Hierarchical attention
        if self.attn_type == "general":
            self.word_linear_in1 = nn.Linear(dim, dim, bias=False)
            self.word_linear_in2 = nn.Linear(dim, dim, bias=False)
            self.word_linear_in3 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in1 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in2 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in3 = nn.Linear(dim, dim, bias=False)
        else:
            raise NotImplementedError

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()    

    def score(self, h_s, type, dist_num, h_t1, h_t2=None, h_t3=None):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t1.size()
            
        h_s_ = h_s.transpose(1, 2)
        dist_lembda1 = 0.5
        dist_lembda2 = 0.5
        
        if type == 'word'and dist_num == "first":
            h_t1_ = self.word_linear_in1(h_t1)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t1, h_s_)
        
        elif type == 'word' and dist_num == "second": 
            h_t1_ = self.word_linear_in2(h_t1)
            h_t2_ = self.word_linear_in2(h_t2)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t2, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_)
            
        elif type == 'word' and dist_num == "third":
            h_t1_ = self.word_linear_in3(h_t1)
            h_t2_ = self.word_linear_in3(h_t2)
            h_t3_ = self.word_linear_in3(h_t3)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            h_t3 = h_t3_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t3, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_) - dist_lembda2 * torch.bmm(h_t2, h_s_)
            
        elif type == 'sent' and dist_num == "first":
            h_t1_ = self.sent_linear_in1(h_t1)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t1, h_s_)

        elif type == 'sent' and dist_num == "second":
            h_t1_ = self.sent_linear_in2(h_t1)
            h_t2_ = self.sent_linear_in2(h_t2)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t2, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_)
       
        elif type == 'sent' and dist_num == "third":
            h_t1_ = self.sent_linear_in3(h_t1)
            h_t2_ = self.sent_linear_in3(h_t2)
            h_t3_ = self.sent_linear_in3(h_t3)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            h_t3 = h_t3_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t3, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_) - dist_lembda2 * torch.bmm(h_t2, h_s_)
            
        else:
            raise NotImplementedError

    def forward(self, word_bank, word_lengths,
                sent_bank, sent_lengths, static_attn, dist_num, source1, source2=None, source3=None):
        
        word_max_len, word_batch, words_max_len, word_dim = word_bank.size()
        sent_max_len, sent_batch, sent_dim = sent_bank.size()
        assert word_batch == sent_batch
        assert words_max_len == sent_max_len
        target_batch, target_dim = source1.size()

        # reshape for compute word score
        # (word_max_len, word_batch, words_max_len, word_dim) -> transpose
        # (word_batch, word_max_len, words_max_len, word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (word_batch, words_max_len, word_max_len, word_dim)
        word_bank = word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            word_batch, words_max_len * word_max_len, word_dim)
        word_align = self.score(word_bank, 'word', dist_num, source1, source2, source3)

        sent_bank = sent_bank.transpose(0, 1).contiguous()
        sent_align = self.score(sent_bank, 'sent', dist_num, source1, source2, source3)        
        
        align = (word_align.view(word_batch, 1, words_max_len, word_max_len) * sent_align.unsqueeze(-1) *\
                     static_attn.unsqueeze(1).unsqueeze(-1)).view(word_batch, 1, words_max_len * word_max_len)
       
        mask = sequence_mask(word_lengths.view(-1), max_len=word_max_len).view(
            word_batch, words_max_len * word_max_len).unsqueeze(1)
        
        align.masked_fill_(~(mask).to(self.device), -float('inf'))
        align_vectors = self.softmax(align) + 1e-20
        c = torch.bmm(align_vectors, word_bank).squeeze(1)
        
        if dist_num == 'first':
            concat_c = torch.cat([c, source1], -1).view(target_batch, target_dim * 2)
        if dist_num == 'second':
            concat_c = torch.cat([c, source2], -1).view(target_batch, target_dim * 2)
        if dist_num == 'third':
            concat_c = torch.cat([c, source3], -1).view(target_batch, target_dim * 2)
                        
        attn_h = self.linear_out(concat_c).view(target_batch, target_dim)
        attn_h = self.tanh(attn_h)

        return attn_h, align_vectors.squeeze(1)
