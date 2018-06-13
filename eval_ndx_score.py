#!/home/heliang/anaconda3/bin/python 
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import time
import getpass
from sys import exit

##  ==========================================================================
##       author : Liang He
##   descrption : eval ndx format score
##                revised from nist sre16 eval tool
##                original author: Omid Sadjadi
##                omid.sadjadi@nist.gov
##      created : 20170104
##      revised : 20180602
## 
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn 
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================

default_seperator = '|'
segment_score_seperator = '|'
model_segment_seperator = '|'

def parse_args():
    """ parse arguments: 
	ndx   : txt, trials (target trials + impostor trials), 
	        format: "model|segment", per line, e.g. 101|abc.wav
	score : txt, scores, 
	        format: "float score", per line, e.g. -1.032
	key   : txt, target trials, 
	        format: "model|segment", per line, e.g. 101|abc.wav
	label : txt, int score, label for each trial, target trial: 1, impostor trial: 0.
	        format: "int score", per line, e.g. 1
   work mode:
	1. provide key, ndx and score (will generate label), maybe very slow
	2. provide label and score, fast
    """
    parser = argparse.ArgumentParser(description="Usage: %prog [options]")
    
    parser.add_argument("--key", type=str, help="key file, target trials")
    parser.add_argument("--label", type=str, help="key file, trial labels")
    parser.add_argument("--ndx", type=str, help="ndx format trial file, e.g. A|B")
    parser.add_argument("--score", type=str, required=True, help="score file, e.g. 0.11534")
 
    args = parser.parse_args()
    return vars(args)

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))


def compute_norm_counts(scores, edges, wghts=None):
    """ computes normalized (and optionally weighted) score counts for the
        bin edges.
    """

    if scores.size > 0:
        score_counts = np.histogram(scores, bins=edges, weights=wghts)[0].astype('f')
        norm_counts = np.cumsum(score_counts)/score_counts.sum()
    else:
        norm_counts = None
    return norm_counts

def compute_pmiss_pfa(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such partitioning).
    """

    tgt_scores = scores[labels==1] # target trial scores
    imp_scores = scores[labels==0] # impostor trial scores

    resol = max([np.count_nonzero(labels==0), np.count_nonzero(labels==1), 1.e6])
    edges = np.linspace(np.min(scores), np.max(scores), resol)

    if weights is not None:
        tgt_weights = weights[labels==1]
        imp_weights = weights[labels==0]
    else:
        tgt_weights = None
        imp_weights = None

    fnr = compute_norm_counts(tgt_scores, edges, tgt_weights)
    fpr = 1 - compute_norm_counts(imp_scores, edges, imp_weights)

    return fnr, fpr

def compute_eer(fnr, fpr):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = ( fnr[x1] - fpr[x1] ) / ( fpr[x2] - fpr[x1] - ( fnr[x2] - fnr[x1] ) )

    return fnr[x1] + a * ( fnr[x2] - fnr[x1] )

def compute_mindcf_sre08(fnr, fpr):
    """ computes the minimum detection cost function 2008 given 
        FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    p_tgt = 0.01
    c_miss = 10
    c_fa = 1
    mindcf_sre08 = compute_c_norm(fnr, fpr, p_tgt, c_miss, c_fa)
    return mindcf_sre08

def compute_mindcf_sre10(fnr, fpr):
    """ computes the minimum detection cost function 2010 given 
        FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    p_tgt = 0.001
    c_miss = 1
    c_fa = 1
    mindcf_sre10 = compute_c_norm(fnr, fpr, p_tgt, c_miss, c_fa)
    return mindcf_sre10

def compute_mindcf_sre12(fnr, fpr):
    """ computes the minimum detection cost function 2012 given 
        FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    p_tgt_1, p_tgt_2 = 0.01, 0.001 
    mindcf_1 = compute_c_norm(fnr, fpr, p_tgt_1)
    mindcf_2 = compute_c_norm(fnr, fpr, p_tgt_2)
    mindcf_sre12 = (mindcf_1 + mindcf_2) / 2
    return [mindcf_sre12, mindcf_1, mindcf_2]

def compute_mindcf_sre14(fnr, fpr):
    """ computes the minimum detection cost function 2014 given 
        FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    ratio = 100
    mindcf_sre14 = compute_c_norm_ratio(fnr, fpr, ratio)
    return mindcf_sre14

def compute_mindcf_sre16(fnr, fpr):
    """ computes the minimum detection cost function 2016 given 
        FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    p_tgt_1, p_tgt_2 = 0.01, 0.005 
    mindcf_1 = compute_c_norm(fnr, fpr, p_tgt_1)
    mindcf_2 = compute_c_norm(fnr, fpr, p_tgt_2)
    mindcf_sre16 = (mindcf_1 + mindcf_2) / 2
    return [mindcf_sre16, mindcf_1, mindcf_2]

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """
    c_det  = min(c_miss * fnr * p_target + c_fa * fpr * ( 1 - p_target ))
    c_def  = min(c_miss * p_target, c_fa * ( 1 - p_target ))
    return c_det/c_def

def compute_c_norm_ratio(fnr, fpr, ratio):
    return min(fnr + ratio * fpr) 
 
def get_lines(fn):
    try:
        fs = open(fn)
        try:
            lines = [line.strip() for line in fs.readlines()]
        finally:
            fs.close()
    except:
        exit("I/O ERROR (%s): %s -- %s" % (errno, strerror, fn))
    return lines

def load_data(fn):
    data = [d for d in get_lines(fn)]
    return data

def convert_to_norm_item(item):
    model = os.path.basename(item.split(default_seperator)[0])
    segment = os.path.basename(item.split(default_seperator)[1])
    return (model + model_segment_seperator + segment)

def norm_key(key_list):
    norm_key_list = []
    for item in key_list:        
        norm_item = convert_to_norm_item(item)
        norm_key_list.append(norm_item)
    return norm_key_list

def load_score_and_make_key(fkey, fndx, fscore):

    score = load_data(fscore)
	
    key_list = load_data(fkey)
    ndx_list = load_data(fndx)
    norm_key_list = norm_key(key_list)
    norm_ndx_list = norm_key(ndx_list)

    label = []
    target_num = 0
    for idx in range(len(norm_ndx_list)):
        if norm_ndx_list[idx] in norm_key_list:
            label.append(int(1))
            target_num = target_num + 1
        else:
            label.append(int(0))

    score = np.hstack(score).astype(np.float)
    label = np.hstack(label).astype(np.int)
    tgt_num = target_num
    imp_num = len(norm_ndx_list) - tgt_num
    
    return [score, label, tgt_num, imp_num]

def load_score_and_label(flabel, fscore):

    label = load_data(flabel)
    score = load_data(fscore)
    score = np.hstack(score).astype(np.float)
    label = np.hstack(label).astype(np.int)
    tgt_num = len(score[label==1])
    imp_num = len(score[label==0])
    
    return [score, label, tgt_num, imp_num]

def eval_ndx_score_label(scores, labels):
    
    [fnr, fpr] = compute_pmiss_pfa(scores, labels)
    eer = compute_eer(fnr, fpr)

    mindcf_sre08 = compute_mindcf_sre08(fnr, fpr)
    mindcf_sre10 = compute_mindcf_sre10(fnr, fpr)
    [mindcf_sre12, mindcf_1, mindcf_2] = compute_mindcf_sre12(fnr, fpr) 
    mindcf_sre14 = compute_mindcf_sre14(fnr, fpr)
    [mindcf_sre16, mindcf_1, mindcf_2] = compute_mindcf_sre16(fnr, fpr)
    
    return eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, mindcf_sre16
    
def eval_ndx_score(fkey, flabel, fndx, fscore):
    
    if os.path.exists(flabel):
        [scores, labels, tgt_num, imp_num] = load_score_and_label(flabel, fscore)
    elif os.path.exists(fkey):
        [scores, labels, tgt_num, imp_num] = load_score_and_make_key(fkey, fndx, fscore)    
        np.savetxt(fndx + ".label", labels, fmt='%d')
    else:
        print("error, please specify label or key+ndx")
        exit(-1)
        
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, 
     mindcf_sre14, mindcf_sre16] = eval_ndx_score_label(scores, labels)
    
    print("------ Aurora Lab ------")
    print("user  :", getpass.getuser())
    print("time  :", get_current_time())
    print("key   :", os.path.basename(fkey))
    print("label :", os.path.basename(flabel))
    print("trial :", os.path.basename(fndx))
    print("score :", os.path.basename(fscore))
    print("total : {0:d}, target: {1:d}, impostor: {2:d}".format(
            tgt_num + imp_num, tgt_num, imp_num))
    print("eer = {0:.2f} %".format(100 * eer))
    print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08))
    print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10))
    print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12))
    print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14))
    print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16))
    print("comment : ")
    print("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(
            100 * eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, 
            mindcf_sre14, mindcf_sre16))


if __name__ == '__main__':

    # args = parse_args()
    # eval_ndx_score(args["key"], args["label"], args["ndx"], args["score"])
	
	eval_ndx_score("nist_sre10_trial_coreext_coreext_key.ndx", 
                "nist_sre10_trial_coreext_coreext_c5_female.ndx.label", 
                "nist_sre10_trial_coreext_coreext_c5_female.ndx", 
                "ivec-dlpp-nist_sre10_trial_coreext_coreext_c5_female.score")
