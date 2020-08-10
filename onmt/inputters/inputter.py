# -*- coding: utf-8 -*-
"""
    Defining general functions for inputters
"""
import glob
import os

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab
from torchtext.data import NestedField

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset
from onmt.utils.logging import logger


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_feats):
    """
    # Flow:0.01
    Args:
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.
        data_type: concat / query / hier
    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    return TextDataset.get_fields(data_type, n_feats)


def load_fields_from_vocab(vocab, data_type="text", n_feats=4):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    fields = get_fields(data_type, n_feats)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
        if isinstance(fields[k], NestedField):
            fields[k].nesting_field.vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)

def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    return levels


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt", "question", "answer"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type=None,
                  data_iter=None, data_path=None,
                  total_token_length=500,
                  src_seq_length=100, src_sent_length=100,
                  seq_length_trunc=0,
                  use_filter_pred=True,
                  feat_name=[]):
    """
    Flow 02: Build src/tgt examples iterator from corpus files, also extract
    number of features.
    
    Flow 07: build object of TextDataset
    """
    # assert data_type is not None
    examples_iter= \
        TextDataset.make_text_examples_nfeats_tpl(
            data_iter, data_path, seq_length_trunc, feat_name)

    dataset = TextDataset(fields, data_type, examples_iter,
                          len(feat_name), total_token_length=total_token_length,
                          src_seq_length=src_seq_length,
                          src_sent_length=src_sent_length,
                          use_filter_pred=use_filter_pred)

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if isinstance(field, NestedField):
        field.nesting_field.vocab = field.vocab
    
def build_vocab(train_dataset, data_type, fields, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        data_type: concat / query / hier
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    dataset = torch.load(train_dataset)
    logger.info(" * reloading %s." % train_dataset)

    for ex in dataset.examples:
        for k in fields:
            val = getattr(ex, k, None)
            if not fields[k].sequential:
                continue   
            if isinstance(fields[k], torchtext.data.NestedField):
                val = [token for tokens in val for token in tokens]
            counter[k].update(val)
  
    #Target and it's features vocab
    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))
    
    for j in range(dataset.n_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    # here we only use src as the source
    # we will copy the vocabulary to others for sharing the source vocab
    #source and it's feature vocab
    _build_field_vocab(fields["src"], counter["src"],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
    logger.info(" * src vocab size: %d." % len(fields["src"].vocab))
    
    for j in range(dataset.n_feats):
        key = "src_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))
    
   #Answer and it's features vocab 
    _build_field_vocab(fields["answer"], counter["answer"],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
    logger.info(" * answer vocab size: %d." % len(fields["answer"].vocab))
    
    for j in range(dataset.n_feats):
        key = "answer_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))
    
    #Question and it's feature vocab
    _build_field_vocab(fields["question"], counter["question"],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
    logger.info(" * question vocab size: %d." % len(fields["question"].vocab))
    
    for j in range(dataset.n_feats):
        key = "question_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    # Merge the input and output vocabularies.
    if share_vocab:
        # `tgt_vocab_size` is ignored when sharing vocabularies
        logger.info(" * merging example vocab...")
        merged_vocab = merge_vocabs(
            [fields["src"].vocab, fields["tgt"].vocab,
             fields["answer"].vocab, fields["question"].vocab],
            vocab_size=src_vocab_size)
        logger.info(" * merged vocab size: %d." % len(merged_vocab))
        fields["src"].vocab = merged_vocab
        fields["src"].nesting_field.vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab
        fields["answer"].vocab = merged_vocab
        fields["question"].vocab = merged_vocab
        

        for j in range(dataset.n_feats):
            logger.info(" * merging feature_%d vocab..." %(j))
            merged_vocab = merge_vocabs(
            [fields["src_feat_"+str(j)].vocab, fields["tgt_feat_"+str(j)].vocab,
             fields["answer_feat_"+str(j)].vocab, fields["question_feat_"+str(j)].vocab],
            vocab_size=src_vocab_size)
            logger.info(" * merged feature_%d vocab size: %d." %(j,  len(merged_vocab)))
            fields["src_feat_"+str(j)].vocab = merged_vocab
            fields["src_feat_"+str(j)].nesting_field.vocab = merged_vocab
            fields["tgt_feat_"+str(j)].vocab = merged_vocab
            fields["answer_feat_"+str(j)].vocab = merged_vocab
            fields["question_feat_"+str(j)].vocab = merged_vocab  
            
    return fields


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Only one inputters.*Dataset, simple!
    pt = opt.data + '.' + corpus_type + '.pt'
    yield _lazy_dataset_loader(pt, corpus_type)


def _load_fields(dataset, data_type, opt, checkpoint, n_feats):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type, n_feats)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    logger.info(' * vocabulary size. source = %d; target = %d' %
                (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')
    qu_features = collect_features(fields, side='question')
    ans_features = collect_features(fields, side='answer')

    return src_features, tgt_features, qu_features, ans_features
