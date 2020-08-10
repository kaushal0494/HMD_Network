"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division

import torch

import onmt.inputters as inputters
import onmt.utils
from onmt.utils.loss import build_loss_compute

from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, fields,
                  optim, data_type, n_feats, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = build_loss_compute(
        model, fields["tgt"].vocab, opt)
    valid_loss = build_loss_compute(
        model, fields["tgt"].vocab, opt, train=False)

    # trunc_size = opt.truncated_decoder  # Badly named...
    # shard_size = opt.max_generator_batches
    # norm_method = opt.normalization
    # grad_accum_count = opt.accum_count

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, n_feats,
                           data_type=data_type,
                           report_manager=report_manager,
                           model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim, n_feats,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 report_manager=None, model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.n_feats = n_feats
        # self.trunc_size = trunc_size
        # self.shard_size = shard_size
        self.data_type = data_type
        # self.norm_method = norm_method
        # self.grad_accum_count = grad_accum_count
        self.report_manager = report_manager
        self.model_saver = model_saver

        # assert grad_accum_count > 0
        # if grad_accum_count > 1:
        #     assert(self.trunc_size == 0), \
        #         """To enable accumulated gradients,
        #            you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, data_iter, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        
        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            for i, batch in enumerate(data_iter("train")):
                self._gradient_accumulation(
                    batch, batch.batch_size, total_stats,
                    report_stats)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                if step % valid_steps == 0:
                    valid_stats = self.validate(data_iter("valid"))
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)

                self._maybe_save(step)
                step += 1
                if step > train_steps:
                    break
        return total_stats

    def validate(self, valid_iter):
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            #with torch.no_grad():
            outputs1, attns1, _, outputs2, attns2, _, outputs3, attns3, _, dis1_sim,\
    dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim = self._forward_prop(batch)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs1, attns1, outputs2, attns2, outputs3, attns3, dis1_sim, dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats


    def _gradient_accumulation(self, batch, normalization, total_stats,
                               report_stats):

        outputs1, attns1, dec_state1, outputs2, attns2, dec_state2, outputs3, attns3, dec_state3, dis1_sim,\
    dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim = self._forward_prop(batch)

        # 3. Compute loss in shards for memory efficiency.
        batch_stats = self.train_loss.compute_loss(
            batch, outputs1, attns1, outputs2, attns2, outputs3, attns3, normalization, dis1_sim, dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim)
        
        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        self.optim.step()

        if dec_state1 is not None:
            dec_state1.detach()
            
        if dec_state2 is not None:
            dec_state2.detach()   
            
        if dec_state3 is not None:
            dec_state3.detach()    

    def _forward_prop(self, batch):
        """forward propagation"""
        # 1, Get all data
        
        # here we use the last word of question as the first word of distractor
        # subsequnetly for features to make the distractor linguistically obey
        # the language model
        
        ques, ques_length = batch.question[0], batch.question[1]    
        tgt_origin = batch.tgt[0][1:,]
        last_ques = torch.stack([ques[:, i][ques_length[i] - 1] for i in range(ques_length.size(0))])
        last_ques = last_ques.unsqueeze(0)
        batch.tgt = list(batch.tgt)
        batch.tgt[0] = torch.cat([last_ques, tgt_origin], 0)
        batch.tgt = tuple(batch.tgt)
               
        src = inputters.make_features(batch, 'src', self.data_type)
        tgt = inputters.make_features(batch, 'tgt', self.data_type)
        question = inputters.make_features(batch, 'question', self.data_type)
        answer = inputters.make_features(batch, 'answer', self.data_type)

        # 2. F-prop all but generator.
        self.model.zero_grad()
        outputs1, attns1, dec_state1, outputs2, attns2, dec_state2, outputs3, attns3, dec_state3, dis1_sim, dis2_sim, \
        dis3_sim, dis12_sim, dis13_sim, dis23_sim  = self.model(src,
                       question, answer, tgt,
                       batch.src[1], batch.src[2],
                       batch.question[1], batch.answer[1], batch.tgt[1])
        
        return outputs1, attns1, dec_state1, outputs2, attns2, dec_state2, outputs3, attns3, dec_state3, dis1_sim,\
    dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim


    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time


    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
