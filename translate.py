#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import ujson as json
import pandas as pd
import csv

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import onmt.opts
from eval.eval import eval


def main(opt):
    translator = build_translator(opt, report_score=False)
    translated = translator.translate(data_path=opt.data,
                                      batch_size=opt.batch_size, 
                                      report_eval_every=opt.report_eval_every)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
