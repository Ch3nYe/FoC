'''
@desc: calculate AUC and ROC metrics for One-to-One comparisons
@usage: python3 cal_auc_roc.py
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pylab
from sklearn.metrics import roc_curve, auc
import json

result_file = "./query_group/crypto_unbalance_XM.scores.json" 

savefig_path = result_file[:-4] + "roc" + ".png"

y_true = []
y_score = []
poolsize = 0 

print(f"[-] reading score file: {result_file}")
with open(result_file, 'r') as jsonf:
    rdata = json.load(jsonf)

# -----  calculate AUC  ----- #
    for rik in rdata:
        poolsize = len(rik) - 1 
        pos_func = rik[0][1]
        for ik in range(len(rik)):
            if ik == 0:
                continue
            x = rik[ik].split(':')
            fid,score = x[0],x[1]
            if str(pos_func) == fid :   
                y_true.append(1)
                y_score.append(float(score))
            else:
                y_true.append(0)
                y_score.append(float(score))
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    auc = auc(fpr,tpr) 

    # -----  Output AUC  ----- #
    print("pos : neg = 1 :",poolsize-1)
    print('auc =', auc)

# # -----  Draw ROC curve  ----- #
lw = 2 
auc = ('%.4f' % auc)
asteria, = plt.plot(fpr, tpr, color='tomato',lw=lw, label='Asteria (area = '+ str(auc) + ')')
plt.legend(loc="lower right",prop={'size':12})  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid( color = 'silver',linestyle='-.',linewidth = 1)
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xlabel('False Positive Rate',fontdict={ 'size' : 14})
plt.ylabel('True Positive Rate',fontdict={ 'size' : 14})
plt.yticks(size = 10)
plt.xticks(size = 10)  
plt.tick_params(width=2,labelsize=12)
plt.show()
plt.savefig(savefig_path)
