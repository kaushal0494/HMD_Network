# coding: utf-8
"""
    Base dataset class and constants
"""
from itertools import chain
import torchtext

import onmt

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class DatasetBase(torchtext.data.Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(DatasetBase, self).__reduce_ex__()

    def load_fields(self, vocab_dict):
        """ Load fields from vocab.pt, and set the `fields` attribute.

        Args:
            vocab_dict (dict): a dict of loaded vocab from vocab.pt file.
        """
        
        fields = onmt.inputters.inputter.load_fields_from_vocab(
            vocab_dict.items(), self.data_type)
        self.fields = dict([(k, f) for (k, f) in fields.items()
                            if k in self.examples[0].__dict__])

    @staticmethod
    def extract_text_features(tokens, side, feat_name_index):
        """
        Args:
            ex: example
            side: flag for source 
            n_feats: num of features
        Returns:
            A sequence of words, a sequence of features, and num of features.
        """        
        if not tokens:
            return [], [], -1

        specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]

        if side == 'src':
            wordss = []
            featuress = []
            for split_tokens in tokens:
                if len(split_tokens) != 0 and split_tokens[0] != '|||': 
                    words = []
                    features = []
                    for split_token in split_tokens:
                        split_token = split_token.split("|")
                        if split_token[0]:
                            words += [split_token[0]]
                            feat_list = [split_token[1:][feat]  for feat in feat_name_index]
                            features += [feat_list]      
                    features = list(zip(*features))
                    wordss.append(words)
                    featuress.append(features)
            
            n_feats_list = list(range(len(feat_name_index)))
            featuress_same = []
            temp_dict = {i:[] for i in range(len(feat_name_index))}
            for sent in featuress:
                for j, fea in enumerate(sent):
                    if j == n_feats_list[j]:
                        temp_dict[j].append(fea)                
            for key in temp_dict:
                featuress_same.append(tuple(temp_dict[key]))   
            return wordss, featuress_same
        else:
            words = []
            features = []
            for split_token in tokens:
                split_token = split_token.split("|")
                if split_token[0]:
                    words += [split_token[0]]
                    feat_list = [split_token[1:][feat]  for feat in feat_name_index]
                    features += [feat_list]  
        
            features = list(zip(*features))
            return tuple(words), features

    # Below are helper functions for intra-class use only.

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        """
        Args:
            seq: an iterator.

        Returns:
            the first thing returned by calling next() on the iterator
            and an iterator created by re-chaining that value to the beginning
            of the iterator.
        """
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
        """
        Args:
            data: the data to be set as the value of the attributes of
                the to-be-created `Example`, associating with respective
                `Field` objects with same key.
            fields: a dict of `torchtext.data.Field` objects. The keys
                are attributes of the to-be-created `Example`.

        Returns:
            the created `Example` object.
        """
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex
