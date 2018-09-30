# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:52:17 2018

@author: tued7001
"""
from library.src.base.cwbook import ConfigWB


class NonCPHWBook(ConfigWB):
    
    def __init__(self, **kwargs):
        ConfigWB.__init__(self, False, '2', modules_run = {'SampleEvaluationInit' : True,
                   'SampleEvaluationFinal' : True,
                   'HotDeck' : True,
                   'PCA' : True,
                   'IW' : True,
                   'FusionEngine' : True,
                   'Create Recips Fused' : True}, **kwargs)
