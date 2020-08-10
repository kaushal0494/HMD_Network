"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters

from onmt.distractor.embeddings import Embeddings
from onmt.distractor.encoder import DistractorEncoder
from onmt.distractor.decoder import HierDecoder
from onmt.distractor.model import DGModel

from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
                         for feat_dict in feature_dicts]

    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]
    
    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    
    n_feats = 0
    for item in checkpoint['vocab']: 
        if item[0].startswith("src_feat"):
            n_feats += 1      
            
    data_type=opt.data_type        
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type, n_feats)

    model_opt = checkpoint['opt']

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the MemModel.
    """
    # Build embedding for encoder.
    src_dict = fields["src"].vocab
    src_feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
    src_embeddings = build_embeddings(model_opt, src_dict, src_feature_dicts) 
    
    
    # Build embedding for decoder.
    tgt_dict = fields["tgt"].vocab
    tgt_feature_dicts = []
    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      tgt_feature_dicts, for_encoder=False)
    
    
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight
    
    encoder = DistractorEncoder(gpu, model_opt.rnn_type,
                                model_opt.word_encoder_type,
                                model_opt.sent_encoder_type,
                                model_opt.question_init_type,
                                model_opt.word_encoder_layers,
                                model_opt.sent_encoder_layers,
                                model_opt.question_init_layers,
                                model_opt.rnn_size, model_opt.dropout,
                                src_embeddings, model_opt.lambda_question,
                                model_opt.lambda_answer, tgt_embeddings)
   

    bidirectional_encoder = True if model_opt.question_init_type == 'brnn' else False
    decoder = HierDecoder(gpu, model_opt.rnn_type, bidirectional_encoder,
                          model_opt.dec_layers, model_opt.rnn_size,
                          model_opt.global_attention,
                          model_opt.dropout,
                          tgt_embeddings)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    model = DGModel(encoder, decoder)

    # Build Generator.
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab)),
        gen_func
    )


    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    """ Build the Model """
    logger.info('Building model...')
    model = build_base_model(model_opt, fields,
                             use_gpu(opt), checkpoint)
    logger.info(model)
    return model
