#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math
from functools import reduce
import ujson as json
import pandas as pd
import csv

from tqdm import tqdm

import torch
import torchtext

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
from eval.eval import eval


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)
    if len(opt.feat_name):
        opt.feat_name = opt.feat_name[0].split("_")
    
    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat", "feat_name",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpuid", "verbose", "fast"
                        ]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, gold_file=opt.target,
                            report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            **kwargs)
    return translator


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    try:
        return len(s1.intersection(s2)) / len(s1.union(s2))
    except ZeroDivisionError:
        return 0.0
    
def eval_save(translated):
    hypothesis = {}
    reference = {}
    j_sim_score = [0, 0, 0]
    same_count = [0, 0, 0]
    for translations in translated:
        pred_list = []
        question_id = str(translations[0].ex_raw.id['file_id']) + '_' + str(translations[0].ex_raw.id['question_id'])
        if question_id not in reference.keys():
             reference[question_id] = [list(translations[0].ex_raw.__dict__['tgt'])]
        else:
             if list(translations[0].ex_raw.__dict__['tgt']) not in reference[question_id]: 
                    reference[question_id].append(list(translations[0].ex_raw.__dict__['tgt']))        
         
        #**************************************** 1st *****************************************
        
        for translation in translations:     
            pred_list.append(translation.pred_sents) 

        pred1, pred2, pred3 = pred_list[0][0], None, None
        for pred in pred_list[1]:
            if jaccard_similarity(pred1, pred) < 0.5:
                if pred2 is None:
                    pred2 = pred
                    break

        if pred2 is None:
            pred2 = pred_list[1][0]

        for pred in pred_list[2]:
            if jaccard_similarity(pred1, pred) < 0.5 and jaccard_similarity(pred2, pred) < 0.5:
                if pred3 is None:
                    pred3 = pred
                    break

        if pred3 is None:
            pred3 = pred_list[2][0]

        hypothesis[question_id] = [pred1, pred2, pred3]
        
        #******************************************** 2nd ****************************************** 
        """
        for translation in translations:     
            pred_list.append(translation.pred_sents)  

        pred1, pred2, pred3 = pred_list[0][0], None, None
        #distractor 2
        for pred in pred_list[1]+ pred_list[0][1:] :
            if jaccard_similarity(pred1, pred) < 0.5:
                if pred2 is None:
                    pred2 = pred
            if pred2 is not None:
                break                   
        if pred2 is None:   
            pred2 = pred_list[1][0]
            
            
        #distractor 3    
        for pred in pred_list[2]+ pred_list[1][1:]+ pred_list[0][1:]:
            if jaccard_similarity(pred1, pred) < 0.5 and jaccard_similarity(pred2, pred) < 0.5:
                if pred3 is None:
                    pred3 = pred
            if pred3 is not None:   
                break        
                    
        if pred3 is None:
            pred3 = pred_list[2][0]

        hypothesis[question_id] = [pred1, pred2, pred3]         
        """
        #************************************* 3rd *******************************************
        """
        for translation in translations:     
            pred_list.append(translation.pred_sents[0])  
        
        if question_id not in hypothesis.keys():     
            hypothesis[question_id] = pred_list
        else:            
            hypothesis[question_id].extend(pred_list)
    
    #extracting three distractors
    for key in hypothesis.keys():
        if len(hypothesis[key]) == 9:
            pred1, pred2, pred3 = hypothesis[key][0], hypothesis[key][3], hypothesis[key][6]
        elif len(hypothesis[key]) == 6:
            pred1, pred2 = hypothesis[key][0], hypothesis[key][3]
            pred3 = None
            for i, pred in enumerate(hypothesis[key]):
                if i != 0 and i != 3 and jaccard_similarity(pred1, pred) < 0.5 and jaccard_similarity(pred2, pred) < 0.5:
                    pred3 = pred
            if pred3 is None:
                pred3 = hypothesis[key][1]
        elif len(hypothesis[key]) == 3:  
            pred1, pred2, pred3 = hypothesis[key][0], hypothesis[key][1], hypothesis[key][2]
        hypothesis[key] = [pred1, pred2, pred3]           
        """
        #******************************************************************************************************* 
    for key in hypothesis.keys(): 
        pred1 = hypothesis[key][0]
        pred2 = hypothesis[key][1]
        pred3 = hypothesis[key][2]
        j_sim_score[0] += jaccard_similarity(pred1 , pred2)
        j_sim_score[1] += jaccard_similarity(pred1 , pred3)
        j_sim_score[2] += jaccard_similarity(pred2 , pred3)

        if pred1 == pred2 and pred2 == pred3 and pred1 == pred3:
            #print(pred1, pred2, pred3)
            same_count[0] += 1
        elif pred1 != pred2 and pred2 != pred3 and pred1 != pred3:
            same_count[2] += 1    
        else:
            same_count[1] += 1

    num_translate = len(hypothesis)
    print("Total Number of records :", num_translate)
    j_sim_score = [ item/float(num_translate) for item in j_sim_score]
    print("Similarity Scores (12, 13, 23) :", j_sim_score)
    print("Same Count (all-3, atleast-2, none) :", same_count) 
    
    with open('pred_dist.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key in hypothesis:
            hyp_sent_list = [' '.join(sent) for sent in hypothesis[key]]
            ref_sent_list = [' '.join(sent) for sent in reference[key]]
            writer.writerow([key, ' | '.join(ref_sent_list), hyp_sent_list[0], hyp_sent_list[1], hyp_sent_list[2] ])
    
    _ = eval(hypothesis, reference)
    

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpuid=-1,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 feat_name = [],
                 ignore_when_blocking=[],
                 use_filter_pred=False,
                 data_type=None,
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 gold_file=None,
                 fast=False):
        self.logger = logger
        self.gpuid = gpuid
        self.cuda = gpuid > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.gold_file = gold_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast
        self.feat_name = feat_name

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  data_path=None,
                  data_iter=None,
                  batch_size=None,
                  max_test_sentences=None,
                  report_eval_every=50):
    
        """
        Translate content of `data_iter` (if not None) or `data_path`
        and get gold scores.

        Note: batch_size must not be None
        Note: one of ('data_path', 'data_iter') must not be None

        Args:
            data_path (str): filepath of source data
            data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert data_iter is not None or data_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          data_type=self.data_type,
                          data_iter=data_iter,
                          data_path=data_path,
                          use_filter_pred=self.use_filter_pred,
                          feat_name = self.feat_name)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        def sort_key(ex):
            """ Sort using length of source sentences. """
            return len(ex.src)
        data_iter = torchtext.data.Iterator(dataset=data,
                                            batch_size=batch_size,
                                            device=cur_device,
                                            train=False, sort=False,
                                            sort_key=sort_key,
                                            repeat=False,
                                            sort_within_batch=False,
                                            shuffle=False)
        
        builder = onmt.translate.TranslationBuilder(
            data, self.data_type, self.fields,
            self.n_best, self.replace_unk, has_tgt=False)
        
        translated = []
        for i, batch in enumerate(tqdm(data_iter)):
            with torch.no_grad():
                batch_data1= self.translate_batch(batch, data, 'first') 
                batch_data2= self.translate_batch(batch, data, 'second')
                batch_data3= self.translate_batch(batch, data, 'third')

            translations1 = builder.from_batch(batch_data1)
            translations2 = builder.from_batch(batch_data2)
            translations3 = builder.from_batch(batch_data3)
            translated.extend([(translations1[0], translations2[0], translations3[0])])
            if i % report_eval_every == 0 and i!=0:
                print("Number of records Processed :", i)
                eval_save(translated)
        print("++++++++++++++ Final Evaluation Scores +++++++++++")
        eval_save(translated) 
        return translated

    def translate_batch(self, batch, data, dis_num):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])
            
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return a.clone().detach().requires_grad_(False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))
        
        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)
        
        
        src = inputters.make_features(batch, 'src', self.data_type)
        tgt = inputters.make_features(batch, 'tgt', self.data_type)
        ques = inputters.make_features(batch, 'question', self.data_type)
        ans = inputters.make_features(batch, 'answer', self.data_type)
        sent_lengths, word_lengths, ques_length,  ans_length = batch.src[1], \
        batch.src[2], batch.question[1], batch.answer[1] 
        
        tgt = tgt[0]
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_length = batch.tgt[1] -1
         
        word_mem_bank, sent_mem_bank, quesinit, static_attn, tgt_state = self.model.encoder(
            src, ques, ans, sent_lengths, word_lengths, ques_length, ans_length, tgt, tgt_length)
       
        enc_state1 = self.model.decoder.init_decoder_state(quesinit)
        enc_state2 = self.model.decoder.init_decoder_state(quesinit)
        enc_state3 = self.model.decoder.init_decoder_state(quesinit)

        # update inputfeed by using question last embedding
        enc_state1.update_state(enc_state1.hidden, enc_state1.hidden[0][-1].unsqueeze(0), enc_state1.coverage)
        enc_state2.update_state(enc_state2.hidden, enc_state2.hidden[0][-1].unsqueeze(0), enc_state2.coverage)
        enc_state3.update_state(enc_state3.hidden, enc_state3.hidden[0][-1].unsqueeze(0), enc_state3.coverage)

        # (2) Repeat src objects `beam_size` times.
        word_mem_bank = var(word_mem_bank.repeat(1, beam_size, 1, 1))
        sent_mem_bank = rvar(sent_mem_bank.data)
        static_attn = static_attn.repeat(beam_size, 1)
        sent_lengths = sent_lengths.repeat(beam_size)
        word_lengths = word_lengths.repeat(beam_size, 1)

        enc_state1.repeat_beam_size_times(beam_size)
        enc_state2.repeat_beam_size_times(beam_size)
        enc_state3.repeat_beam_size_times(beam_size)
        
        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))
            #print(inp)
            # Run one step.
            dec_h1, dec_out1, dec_states1, attn1, dec_h1, dec_out2, dec_states2, attn2, dec_h1, dec_out3, dec_states3, attn3 =\
            self.model.decoder(inp, word_mem_bank, sent_mem_bank,
                                   enc_state1, enc_state2, enc_state3, word_lengths,
                                   sent_lengths, static_attn)
           
            if dis_num == 'first':
                dec_out = dec_out1
                dec_states = dec_states1
                attn = attn1
            elif dis_num == 'second':
                dec_out = dec_out2
                dec_states = dec_states2
                attn = attn2
            elif dis_num == 'third':
                dec_out = dec_out3
                dec_states = dec_states3
                attn = attn3

            dec_out = dec_out.squeeze(0)
            # (b) Compute a vector of batch x beam word scores.
            out = self.model.generator.forward(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab
            beam_attn = unbottle(attn["std"])

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j], beam_attn.data[:, j, :])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret              
        
    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret


