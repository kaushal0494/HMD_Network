# -*- coding: utf-8 -*-
"""Define word-based embedders."""

from collections import Counter
from itertools import chain
import io
import codecs
import sys
import ujson as json

import torch
import torchtext

from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD,
                                         PAD_WORD, BOS_WORD, EOS_WORD)
from onmt.utils.misc import aeq
from onmt.utils.logging import logger


class TextDataset(DatasetBase):
    """ Flow8: -Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, data_type, examples_iter,
                 n_feats=0, total_token_length=0,
                 src_seq_length=0, src_sent_length=0,
                 use_filter_pred=True):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_feats = n_feats # num of features

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)

        keys = ['id', 'total_tokens']
        for key in fields.keys():
            keys.append(key)       
            
        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]

        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            out_examples.append(example)
        
        logger.info("{} Example before filter".format(len(out_examples)))

        def filter_pred(example):
            """ ? """
            # filter num_sent > thres; max_len_sent > thres; total_token > thres
            if example.total_tokens >= total_token_length:
                return False
            elif len(example.src) >= src_sent_length:
                return False
            elif max(list(len(tokens) for tokens in example.src)) >= src_seq_length:
                return False
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

        logger.info("{} Example after filter".format(len(self.examples)))

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src) + len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, feat_name):
        #Flow-03: Loading data and Creating featues and 
        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        #examples_nfeats_iter = \
        return TextDataset.make_examples(text_iter, truncate, feat_name)

        #first_ex = next(examples_nfeats_iter)
        #num_feats = first_ex[1]
        
        # Chain back the first element - we only want to peek it.
        #examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        #examples_iter = (ex for ex in examples_nfeats_iter)

        #return  examples_iter 

    @staticmethod
    def make_examples(text_iter, truncate, feat_name):
        # Creating dic of  the word and its features for every components of dataset
        # ******************** Hard Coding *****************
        feat_name_index = []
        if 'pos' in feat_name:
            feat_name_index.append(0)
        if 'ner' in feat_name:
            feat_name_index.append(1)
        if 'dep' in feat_name:
            feat_name_index.append(2)
        if 'lemma' in feat_name:
            feat_name_index.append(3)   
                
        # *********************** End *************************        
            
        for i, line in enumerate(text_iter):
            ex = json.loads(line)
            
            src_words, src_feats = \
                TextDataset.extract_text_features(ex['sent'], 'src', feat_name_index)
            ans_words, ans_feats = \
               TextDataset.extract_text_features(ex['answer_text'], 'ans', feat_name_index)
            qu_words, qu_feats = \
                TextDataset.extract_text_features(ex['question'], 'ques', feat_name_index)
            tgt_words, tgt_feats = \
                TextDataset.extract_text_features(ex['distractor'], 'tgt', feat_name_index)

            example_dict = {
                'src': src_words,
                'answer': ans_words,
                'question': qu_words,
                'tgt': tgt_words,
                "indices": i,
                'id': ex['id'],
                'total_tokens': len(ans_words) + len(qu_words) + len(tgt_words) \
                                           + sum(list(len(tokens) for tokens in src_words))
            }

            #Adding Features
            example_dict.update(("src_feat_" + str(j), f)
                                    for j, f in enumerate(src_feats))
            example_dict.update(("answer_feat_" + str(j), f)
                                    for j, f in enumerate(ans_feats))
            example_dict.update(("question_feat_" + str(j), f)
                                    for j, f in enumerate(qu_feats))
            example_dict.update(("tgt_feat_" + str(j), f)
                                    for j, f in enumerate(tgt_feats))
            
            yield example_dict

    @staticmethod
    def make_text_iterator_from_file(path):
        #Flow-04: Loading data as generator object
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(data_type, n_feats):
        """
        Creating the different type of torchtext object for different components of dataset i.e. src, question, answer, tgt and indices 
        
        """
        fields = {}

        # we only use this single fields in vocab

        #source and its features
        fields["src"] = torchtext.data.NestedField(
            torchtext.data.Field(pad_token=PAD_WORD), include_lengths=True)
        
        for j in range(n_feats):
            fields["src_feat_" + str(j)] = torchtext.data.NestedField( torchtext.data.Field(pad_token=PAD_WORD))

            
        #Question and its features    
        fields["question"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        
        for j in range(n_feats):
            fields["question_feat_" + str(j)] = \
                 torchtext.data.Field(pad_token=PAD_WORD)
        
        
        #Answer and its features
        fields["answer"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        
        for j in range(n_feats):
            fields["answer_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

            
        #Target and its features    
        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD, include_lengths=True)
        
        for j in range(n_feats):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)
        
        #indices
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            ex = json.loads(cf.readline())
            _, _, num_feats = TextDataset.extract_text_features(ex[side], side)
        return num_feats




