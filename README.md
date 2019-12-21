# G-MmDAE
--The code for the paper: "Neighbourhood Structure Preserving Cross-modal Embedding for Video Hyperlinking"--  
—-Version 1.0—-   
--Updated: 21/12/2019--  

# Requirements to run the experiments  
Python 2  
Pytorch  [http://pytorch.org/]  
Numpy  
Sklearn  [http://scikit-learn.org/stable/]  

# Used datasets  
MST-VTT: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf  
Blip10000: http://doras.dcu.ie/17922/2/MMSys2013BBlip10000.pdf  

# Usage  
python train_*.py  
python test.py  
for computing the Gaussian parameter sigma, go to https://lvdmaaten.github.io/tsne/  

# Evaluation
'retrieval_lnk.py' in './evaluate' is for creating the linking runs  
please download the code from https://github.com/robinaly/sh_eval to ./evaluate  

# Author Email:
haoyanbin@hotmail.com  
