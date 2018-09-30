# -*- coding: utf-8 -*-
"""
Created on Tue May 15 04:52:04 2018

@author: tued7001
"""

from library.src.base.cwbook import ConfigWB

class CPHWBook(ConfigWB):
    
    def __init__(self, **kwargs):
        ConfigWB.__init__(self, True, '1', modules_run = {'SampleEvaluationInit' : True,
                   'SampleEvaluationFinal' : True,
                   'HotDeck' : True,
                   'PCA' : True,
                   'IW' : True,
                   'FusionEngine' : True,
                   'Create Recips Fused' : True}, **kwargs)

