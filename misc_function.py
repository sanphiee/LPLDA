#!/home/heliang/anaconda3/bin/python 
# -*- coding: utf-8 -*-

import csv
import os
import numpy as np
from sys import exit

##  ==========================================================================
##       author : Liang He
##   descrption : misc function for sre10 demo
##      created : 20180613
##      revised : 
## 
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn 
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================

default_seperator = '|'

def load_ivectors(filename):
    """Loads ivectors

    Parameters
    ----------
    filename : string
        Path to ivector files (e.g. dev_ivectors.csv)

    Returns
    -------
    ids : list
        List of ivectorids
    ivectors : array, shaped('n_ivectors', 600)
        Array of ivectors for each ivectorid
    """
    
    ids = []
    ivectors = []

    for row in csv.DictReader(open(filename, 'r')):
        ids.append( row['ivectorid'] )
        ivectors.append( np.fromstring(row['values[600]'], count=600, sep=' ',
                                       dtype=np.float32) )

    return ids, np.vstack( ivectors )
	
def get_lines(fn):
    try:
        fs = open(fn)
        try:
            lines = [line.strip() for line in fs.readlines()]
        finally:
            fs.close()
    except:
        exit("I/O ERROR : %s" % (fn))
    return lines

def load_data(fn):
    data = [d for d in get_lines(fn)]
    return data

def convert_to_norm_item(item):
    model = os.path.basename(item.split(default_seperator)[0])
    segment = os.path.basename(item.split(default_seperator)[1])
    return (model + default_seperator + segment)

def norm_ndx(key_list):
    norm_ndx_list = []
    for item in key_list:        
        norm_item = convert_to_norm_item(item)
        norm_ndx_list.append(norm_item)
    return norm_ndx_list

def label_str_to_int(label_str):
    """label, string to int

    Parameters
    ----------
    filename : string label

    Returns
    -------
    int label
    """
    
    label_dict = {}
    label_int = []
    for item in label_str:
        if item not in label_dict.keys():
            label_dict[item] = len(label_dict) + 1
        label_int.append(label_dict[item])
    
    return np.array(label_int)

def generate_lambda_label(flambda_ivec, flambda_ndx, flabel):
    
    lambda_list = load_data(flambda_ndx)
    norm_lambda_list = norm_ndx(lambda_list)
    
    lambda_map = dict(
            [(i.split(default_seperator)[1], i.split(default_seperator)[0]) 
            for i in norm_lambda_list])
    
    dev_ids, dev_ivec = load_ivectors(flambda_ivec)
    
    dev_label = []
    for dev_id in dev_ids:
        dev_label.append(lambda_map[dev_id])
    
    dev_int_label = label_str_to_int(dev_label)
    max_label = max(dev_int_label)
    
    for i in range(max_label):
        label_count = len(dev_int_label[dev_int_label == i])
        if label_count < 6:
            dev_int_label[dev_int_label == i] = -2
    
    np.savetxt(flabel, dev_int_label, fmt='%d')

def load_score_and_make_key(fkey, fndx):

    key_list = load_data(fkey)
    ndx_list = load_data(fndx)
    norm_key_list = norm_ndx(key_list)
    norm_ndx_list = norm_ndx(ndx_list)

    label = np.zeros(len(norm_ndx_list), dtype=np.int)
    target_num = 0
    for idx in range(len(norm_ndx_list)):
        if norm_ndx_list[idx] in norm_key_list:
            label[idx] = 1
            target_num = target_num + 1

    tgt_num = target_num
    imp_num = len(norm_ndx_list) - tgt_num
    
    return [label, tgt_num, imp_num]

def generate_trial_label(fndx, fkey, flabel):
    
    [label, tgt_num, imp_num] = load_score_and_make_key(fkey, fndx)
    print('tgt_num %d, imp_num %d' % (tgt_num, imp_num))
    np.savetxt(flabel, label, fmt='%d')
	
	
def norm_ndx_list(ndx_list):
    """ get norm ndx format list: 
	 ndx_list : txt, trials (target trials + impostor trials), 
	            format: "model|segment", per line, e.g. 101|abc.wav
    return :
        ndx     : normalized ndx format list, i.e. remove path
        model   : sorted model name, ascend
        segment : sorted segment name, ascend
    """
    norm_ndx_list = []
    model_list = []
    segment_list = []
    
    for item in ndx_list:        
        model = os.path.basename(item.split(default_seperator)[0])
        segment = os.path.basename(item.split(default_seperator)[1])
        norm_item = model + default_seperator + segment
        model_list.append(model)
        segment_list.append(segment)
        norm_ndx_list.append(norm_item)
    
    unique_model = list(set(model_list))
    unique_segment = list(set(segment_list))
    norm_ndx_list.sort()
    unique_model.sort()
    unique_segment.sort()
    
    return norm_ndx_list, unique_model, unique_segment

def generate_trial_mask(fndx, fndx_sort, fmask):
    ndx_list = load_data(fndx)
    [normed_ndx_list, model_list, segment_list] = norm_ndx_list(ndx_list)
    mask_label = np.zeros(len(model_list) * len(segment_list), dtype=np.int)
    
    model_pos = dict([(model_list[i], i) 
    for i in range(len(model_list))])
    segment_pos = dict([(segment_list[i], i) 
    for i in range(len(segment_list))])
    
    for item in normed_ndx_list:
        model = item.split(default_seperator)[0]
        segment = item.split(default_seperator)[1]
        model_index = model_pos[model]
        segment_index = segment_pos[segment]
        ndx_index = model_index * len(segment_list) + segment_index
        mask_label[ndx_index] = 1
    
    print('model size %d, segment size %d' % 
          (len(model_list), len(segment_list)))
    np.savetxt(fmask, mask_label, fmt='%d')
    np.savetxt(fndx_sort, normed_ndx_list, fmt='%s')