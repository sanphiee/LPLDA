#*- coding:UTF-8 -*-
"""
##  ==========================================================================
##
##       author : Liang He, heliang@mail.tsinghua.edu.cn
##                Xianhong Chen, chenxianhong@mail.tsinghua.edu.cn
##   descrption : sre10 demo
##                comparison of LDA and LPLDA
##                LDA: linear discriminant analysis
##                LPLDA: local pairwise linear discriminant analysis
##      created : 20180612
## last revised : 
##
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================
"""

import numpy as np
import eval_ndx_score
import misc_function
import LDA
import LPLDA
import sys
	
def TestPrepare():
    
    data_path = './data/'
    
    misc_function.generate_lambda_label(
            data_path + "sre050608_swb_male_lambda_ivec.csv", 
            data_path + "sre050608_swb_male_lambda.ndx", 
            data_path + "sre050608_swb_male_lambda.ndx.label")
    
    misc_function.generate_trial_mask(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx', 
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort', 
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.mask')
    
    misc_function.generate_trial_label(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort', 
            data_path + 'nist_sre10_trial_coreext_coreext_key.ndx', 
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.label')

def TestCosine():

    print("begin test cosine ...")

    data_path = './data/'
    
    ## load ivec
    model_ids, model_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_train_ivec.csv')
    test_ids, test_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_test_ivec.csv')
    
    ## load mask and label
    trial_mask = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.mask')
    trial_label = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.label')
    
    ## length norm
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    
    ## cosine
    score_matrix = np.dot(np.asarray(model_ivec), np.asarray(test_ivec.T))
    score = np.asarray(score_matrix.reshape(-1,1))
    
    ## trial score
    trial_score = score[trial_mask == 1]
    
    ## eval
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, 
     mindcf_sre16] = eval_ndx_score.eval_ndx_score_label(
     trial_score, trial_label)
    
    with open('./results/sre10_demo_result.txt','a+') as f:
        print("------ Aurora Lab ------", file = f)
        print("eer = {0:.2f} %".format(100 * eer), file = f)
        print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08), file = f)
        print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10), file = f)
        print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12), file = f)
        print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14), file = f)
        print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16), file = f)
        print("comment : cosine", file = f)
        print("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(
                100 * eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, 
                mindcf_sre14, mindcf_sre16), file = f)
    
def TestLDA():

    print("begin test LDA ...")

    data_path = './data/'
    lda_dim = 150
    
    ## load ivec
    dev_ids, dev_ivec = misc_function.load_ivectors(
            data_path+'sre050608_swb_male_lambda_ivec.csv')
    model_ids, model_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_train_ivec.csv')
    test_ids, test_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_test_ivec.csv')
    
    ## load mask and label
    lambda_label = np.loadtxt(
            data_path + 'sre050608_swb_male_lambda.ndx.label')
    trial_mask = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.mask')
    trial_label = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.label')
    
    dev_ivec = dev_ivec[lambda_label > -1]
    lambda_label = lambda_label[lambda_label > -1]
    
    # remove mean
    m = np.mean(dev_ivec, axis=0)
    dev_ivec = dev_ivec - m
    model_ivec = model_ivec - m
    test_ivec = test_ivec - m
    
    ## LDA
    lda = LDA.LinearDiscriminantAnalysis(n_components=lda_dim)
    lda.fit(np.asarray(dev_ivec), np.asarray(lambda_label))
    model_ivec = lda.transform(np.asarray(model_ivec))
    test_ivec = lda.transform(np.asarray(test_ivec))

    ## length norm
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
        
    ## cosine
    score_matrix = np.dot(np.asarray(model_ivec), np.asarray(test_ivec.T))
    score = np.asarray(score_matrix.reshape(-1,1))
    
    ## trial score
    trial_score = score[trial_mask == 1]
    
    ## eval
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, 
     mindcf_sre16] = eval_ndx_score.eval_ndx_score_label(
     trial_score, trial_label)
    
    with open('./results/sre10_demo_result.txt','a+') as f:
        print("------ Aurora Lab ------", file = f)
        print("eer = {0:.2f} %".format(100 * eer), file = f)
        print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08), file = f)
        print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10), file = f)
        print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12), file = f)
        print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14), file = f)
        print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16), file = f)
        print("comment : LDA", file = f)
        print("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(
                100 * eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, 
                mindcf_sre14, mindcf_sre16), file = f)

def TestLPLDA():

    print("begin test LPLDA ...")

    data_path = './data/'
    lda_dim = 150
    
    ## load ivec
    dev_ids, dev_ivec = misc_function.load_ivectors(
            data_path+'sre050608_swb_male_lambda_ivec.csv')
    model_ids, model_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_train_ivec.csv')
    test_ids, test_ivec = misc_function.load_ivectors(
            data_path + 'nist_sre10_c5_coreext_male_test_ivec.csv')
    
    ## load mask and label
    lambda_label = np.loadtxt(
            data_path + 'sre050608_swb_male_lambda.ndx.label')
    trial_mask = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.mask')
    trial_label = np.loadtxt(
            data_path + 'nist_sre10_trial_coreext_coreext_c5_male.ndx.sort.label')

    dev_ivec = dev_ivec[lambda_label > -1]
    lambda_label = lambda_label[lambda_label > -1]
    
    # remove mean
    m = np.mean(dev_ivec, axis=0)
    dev_ivec = dev_ivec - m
    model_ivec = model_ivec - m
    test_ivec = test_ivec - m

    ## LPLDA
    lda = LPLDA.LocalPairwiseLinearDiscriminantAnalysis(
        n_components=lda_dim)
    lda.fit(np.asarray(dev_ivec), np.asarray(lambda_label))
    model_ivec = lda.transform(np.asarray(model_ivec))
    test_ivec = lda.transform(np.asarray(test_ivec))
    
    ## length norm
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    
    ## cosine
    score_matrix = np.dot(np.asarray(model_ivec), np.asarray(test_ivec.T))
    score = np.asarray(score_matrix.reshape(-1,1))
    
    ## trial score
    trial_score = score[trial_mask == 1]
    
    ## eval
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, 
     mindcf_sre16] = eval_ndx_score.eval_ndx_score_label(
     trial_score, trial_label)
    
    with open('./results/sre10_demo_result.txt','a+') as f:
        print("------ Aurora Lab ------", file = f)
        print("eer = {0:.2f} %".format(100 * eer), file = f)
        print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08), file = f)
        print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10), file = f)
        print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12), file = f)
        print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14), file = f)
        print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16), file = f)
        print("comment : LPLDA", file = f)
        print("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(
                100 * eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, 
                mindcf_sre14, mindcf_sre16), file = f)
    
if __name__=='__main__':
    
    TestPrepare()
    TestCosine()
    TestLDA()
    TestLPLDA()
    
    print('done')
    
