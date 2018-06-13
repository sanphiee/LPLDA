# LPLDA
Local Pairwise Linear Discriminant Analysis

This is a demo for comparing LDA and LPLDA on NIST SRE2010 c5 coreext-coreext male condition.

The same version to Code Ocean.

environment : anaconda3, python3, require sklearn.

1. scripts \
a) sre10_demo.py, main function \
b) LDA.py, linear discriminant analysis \
c) LPLDA.py, local pairwise linear discriminant analysis \
d) misc_function.py, misc functions, e.g. load ivectors, process lists and so on. \
e) eval_ndx_score.py, eval score

2. data \
A. GMM-2048-Diag/ivectors, 600 dimension, extracted by Aurora3 Project developed by Aurora Lab, Dept.E.E., Tsinghua University \
a) train ivectors, csv format file:  \
   nist_sre10_c5_coreext_male_train_ivec.csv \
b) test ivectors, csv format file:  \
   nist_sre10_c5_coreext_male_test_ivec.csv \
c) lambda ivectors (for training LDA, LPLDA), csv format file:  \
   sre050608_swb_male_lambda_ivec.csv \
B. list \
a) train, ndx format file: since the train ivectors is named by model_id, this file is unnecessary. \
   nist_sre10_train_coreext_male.ndx \
b) trial, ndx format file: \
   nist_sre10_trial_coreext_coreext_c5_male.ndx \
c) lambda, ndx format file: for training LDA, LPLDA \
   sre050608_swb_male_lambda.ndx \
d) key, ndx format file: all target trial \
   nist_sre10_trial_coreext_coreext_key.ndx

3. Procedure \
We evaluate the LDA and LPLDA based on extracted ivectors.

4. Results \
------ Aurora Lab ------ \
eer = 6.75 % \
mindcf_sre08 = 0.2710 \
mindcf_sre10 = 0.6121 \
mindcf_sre12 = 0.5274 \
mindcf_sre14 = 0.4437 \
mindcf_sre16 = 0.4722 \
comment : cosine \
6.7509, 0.2710, 0.6121, 0.5274, 0.4437, 0.4722 \
------ Aurora Lab ------ \
eer = 3.67 % \
mindcf_sre08 = 0.1727 \
mindcf_sre10 = 0.4417 \
mindcf_sre12 = 0.3775 \
mindcf_sre14 = 0.3140 \
mindcf_sre16 = 0.3348 \
comment : LDA \
3.6652, 0.1727, 0.4417, 0.3775, 0.3140, 0.3348 \
------ Aurora Lab ------ \
eer = 3.35 % \
mindcf_sre08 = 0.1330 \
mindcf_sre10 = 0.3486 \
mindcf_sre12 = 0.2946 \
mindcf_sre14 = 0.2411 \
mindcf_sre16 = 0.2604 \
comment : LPLDA \
3.3478, 0.1330, 0.3486, 0.2946, 0.2411, 0.2604


Our paper reported results: \
       EER[%] MDCF10 \
cosine 6.75   0.612 \
LDA    3.76   0.458 \
LPLDA  2.97   0.355 

This code's results: \
       EER[%] MDCF10 \
cosine 6.75   0.612 \
LDA    3.67   0.442 \
LPLDA  3.35   0.348 

Why different ? Our reported results are realized by C++ code (Aurora3 Project). 
This python version is a revised one based on the C++ code.
There are several possible explanations for this difference, 
e.g. float vs double, realization of eigen decomposition algorithm.
It's hard to make them same. 
In either case, LPLDA is significantly better than LDA in this test.

Results of our C++ code \
------ Aurora Lab ------ \
user  : heliang \
time  : 2018-03-23 20:17:12 \
key   : nist_sre10_trial_coreext_coreext_key.ndx \
trial : nist_sre10_trial_coreext_coreext_c5_male.ndx \
score : ivec-nist_sre10_trial_coreext_coreext_c5_male.score \
total : 179338, target: 3465, impostor: 175873 \
eer = 6.75 % \
mindcf_sre08 = 0.2710 \
mindcf_sre10 = 0.6121 \
mindcf_sre12 = 0.5274, mindcf1 = 0.4427, mindcf2 = 0.5016 \
mindcf_sre14 = 0.4437 \
mindcf_sre16 = 0.4722, mindcf1 = 0.4427, mindcf2 = 0.5016 \
comment : cosine 
 
------ Aurora Lab ------ \
user  : heliang \
time  : 2018-03-24 11:33:12 \
key   : nist_sre10_trial_coreext_coreext_key.ndx \
trial : nist_sre10_trial_coreext_coreext_c5_male.ndx \
score : ivec-lda-nist_sre10_trial_coreext_coreext_c5_male.score \
total : 179338, target: 3465, impostor: 175873 \
eer = 3.76 % \
mindcf_sre08 = 0.1586 \
mindcf_sre10 = 0.4587 \
mindcf_sre12 = 0.3799, mindcf1 = 0.3012, mindcf2 = 0.3494 \
mindcf_sre14 = 0.3018 \
mindcf_sre16 = 0.3253, mindcf1 = 0.3012, mindcf2 = 0.3494 \
3.7641, 0.1586, 0.4587, 0.3799, 0.3018, 0.3253 \
comment : LDA 
 
------ Aurora Lab ------ \
user  : heliang \
time  : 2018-03-24 11:28:28 \
key   : nist_sre10_trial_coreext_coreext_key.ndx \
trial : nist_sre10_trial_coreext_coreext_c5_male.ndx \
score : ivec-dlpp-nist_sre10_trial_coreext_coreext_c5_male.score \
total : 179338, target: 3465, impostor: 175873 \
eer = 2.97 % \
mindcf_sre08 = 0.1264 \
mindcf_sre10 = 0.3558 \
mindcf_sre12 = 0.2970, mindcf1 = 0.2381, mindcf2 = 0.2775 \
mindcf_sre14 = 0.2387 \
mindcf_sre16 = 0.2578, mindcf1 = 0.2381, mindcf2 = 0.2775 \
comment : LPLDA 

He Liang, Tsinghua University
June 13, 2018