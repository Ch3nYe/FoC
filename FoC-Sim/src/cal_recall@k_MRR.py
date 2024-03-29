'''
@desc: calculate recall@k and MRR metrics for One-to-Many comparisons
@usage: python3 cal_auc_roc.py
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pylab
from sklearn.metrics import roc_curve,auc
import json

result_file = "./query_group/crypto_unbalance_XM.scores.json" 

savefig_path = result_file[:-4] + "recall@k" + ".png"

recall_list = []
recall_1 = 0 
recall_10 = 0
recall_50 = 0
poolsize = 0

print(f"[-] reading score file: {result_file}")
with open(result_file, 'r') as jsonf:
    rdata = json.load(jsonf)

# -----  calculate MRR  ----- #
    MRR = 0
    for rik in rdata:
        poolsize = len(rik) - 1
        pos_func = rik[0][1]
        for ik in range(len(rik)):
            if ik == 0:
                continue
            x = rik[ik].split(':')
            if str(pos_func) == x[0]:   # x[0]
                MRR += 1/ik
                break
    print("MRR:",MRR/len(rdata))

# -----  calculate MRR10  ----- #
    MRR10 = 0
    for rik in rdata:
        poolsize = len(rik) - 1
        pos_func = rik[0][1]
        for ik in range(len(rik)):
            if ik == 0:
                continue
            if ik == 11:   
                break
            x = rik[ik].split(':')
            if str(pos_func) == x[0]:   # x[0]
                MRR10 += 1/ik
                break
    print("MRR10:",MRR10/len(rdata))

# -----  calculate Recall@K  ----- #
    for topk in range(51):
        if topk == 0:
            continue
        recall = 0
        for rik in rdata:
            pos_func = rik[0][1]
            for ik in range(topk+1) :
                if ik == 0 :
                    continue
                x = rik[ik].split(':')
                if str(pos_func) == x[0]:
                    recall += 1
                    break
        #print(recall)
        recall = recall/len(rdata)
        if topk == 1:
            recall_1 = recall
        if topk == 10:
            recall_10 = recall
        if topk == 50:
            recall_50 = recall            
        recall_list.append(recall)
    #print(recall_list)
    print("recall@1 =",recall_1)
    print("recall@10 =",recall_10)
    print("recall@50 =",recall_50)
    print("poolsize =",poolsize)

# -----  Draw Recall@K curve  ----- #
lw = 2 
x = list(range(1, 51, 1))
asteria, = plt.plot(x, recall_list,color='tomato',lw=lw,linestyle='--')
plt.legend(handles=[asteria], labels=['Asteria'], loc="lower right",prop={'size':14})  
plt.grid( color = 'silver',linestyle='-.',linewidth = 1)
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xlim([0, 50])
plt.ylim([0, 1.0])
plt.xlabel('Number of results retrieved(K)',fontdict={'size' : 14})
plt.ylabel('Recall@K',fontdict={'size': 14})
plt.yticks(size = 10)
plt.xticks(size = 10)  
plt.tick_params(width=2,labelsize=12)
plt.show()
plt.savefig(savefig_path)