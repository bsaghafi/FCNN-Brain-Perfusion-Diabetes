# Date: 07/14/2017
# Author: Behrouz Saghafi
import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import timeit

start=timeit.default_timer()
df = pd.read_csv('AADHSMind_JJ_Data20151124_RelevantColumns_v2ColorCoded_CACandCRPadded_JUST_THE_SUBJECTS_WITH_GOOD_ImagesDataColsEU_EV_FK_FP_FW.csv')
gfr = df['ckd_gfr'].values
# print gfr.dtype
# print gfr.shape
idx=np.nonzero(~np.isnan(gfr))
N=len(gfr)
gfr = np.reshape(gfr, (N,1))
# print idx
# GFR=gfr[idx]
# print GFR.shape

id = df['id'].values
id = np.reshape(id, (N,1))
date=df['date11'].values
date = np.reshape(date, (N,1))
arr = np.concatenate((id, date, gfr), axis=1)
arr=arr[idx]
print arr.shape
print arr.shape[0]

# CBF csv:
df2 = pd.read_csv('ANSIR_CBF_VBM8_AALdata_GM.csv')

MR_ID=[]
for i in range(arr.shape[0]):
    ID=arr[i,0]
    if ID<100000:
        name = 'AADHS' + '0' + str(int(arr[i, 0])) + '_' + str(int(arr[i, 1]))
    else:
        name = 'AADHS' + str(int(arr[i, 0])) + '_' + str(int(arr[i, 1]))
    # print name
    MR_ID.append(name)

MR_ID = np.reshape(MR_ID, (arr.shape[0], 1))
arr = np.concatenate((arr, MR_ID), axis=1)

dfa = pd.DataFrame(data=arr, columns=['id', 'data', 'gfr', 'MR_ID'])
combine = pd.merge(dfa, df2, how='inner', on='MR_ID')
index=np.isnan(combine.iloc[:, 8:].values)
# print np.nonzero(index)
# print ('nancount=',np.sum(index))
combine.to_csv('combine.csv')
C = np.array(combine)
C=Imputer(missing_values='NaN', strategy='median', axis=1, verbose=0, copy=True)
# print('dimensions are:',C.shape)
# Cv=combine.iloc[:, :].values
# print Cv
C = C.fit_transform(combine.iloc[:, 8:].values)
C=np.array(C)
y = combine['gfr']
x1 = combine['Hippocampus_L']
x2 = combine['Hippocampus_R']
n=len(y)

# print('****Uni-variable univariate analysis******')
y=np.asarray(y,dtype=np.float64)
# print y.shape
# print y.dtype
# print x1.shape
# print x1.dtype
# x1 = sm.add_constant(x1)
# est = sm.OLS(y, x1)
# est2 = est.fit()
# print est2.rsquared
# print est2.f_pvalue
# print est2.pvalues[1]
# print(est2.summary())

# x2 = sm.add_constant(x2)
# est = sm.OLS(y, x2)
# est2 = est.fit()
# print(est2.summary('R-squared'))
# AAL=pd.read_excel('AAL.xlsx')
AAL=pd.read_csv('AAL.csv')
roi_n=AAL.iloc[:, 0].values
roi_names=AAL.iloc[:, 1].values
R2=[]
pval=[]
# for i in range(116):
#     x = C[:,i]
#     x=np.reshape(x,(n,1))
#     x = sm.add_constant(x)
#     est = sm.OLS(y, x)
#     est2 = est.fit()
#     R2.append(est2.rsquared)
#     pval.append(est2.pvalues[1])
#     if est2.rsquared>0.01 and est2.pvalues[1]<0.05:
#         print ('structure_name=',roi_names[i],'R2=',est2.rsquared,'pvalue=',est2.pvalues[1])
    # print('ROI=',i,'%R2=',R2[i,0])

# fig1, ax=plt.subplots()
# plt.bar(np.arange(0, 116, 1),R2)
# ax.set_xticks(np.arange(0, 116, 1))
# ax.set_xticklabels(roi_names, rotation='vertical')
# ax.set_title('predicting duration')
# ax.set_xlabel('AAL ROI')
# ax.set_ylabel('R2')
# plt.tight_layout()
# plt.show()
#
# fig2, ax=plt.subplots()
# plt.bar(np.arange(0, 116, 1),pval)
# ax.set_xticks(np.arange(0, 116, 1))
# ax.set_xticklabels(roi_names, rotation='vertical')
# ax.set_title('predicting duration')
# ax.set_xlabel('AAL ROI')
# ax.set_ylabel('pvalue')
# plt.tight_layout()
# plt.show()

print('*************Multi-variable univariate analysis**************')
y=np.asarray(y,dtype=np.float64)
X=[]
for i in range(116):
    X.append(C[:,i])
X=np.reshape(X,(n,116))
X=MinMaxScaler().fit_transform(X)
print('X_shape=',X.shape)
# np.save('X_train_gfr',X)
X = sm.add_constant(X)
est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())
# print est2.pvalues[est2.pvalues<0.01]
# print est2.rsquared
# print est2.params

# AAL=pd.read_excel('AAL.xlsx')
AAL=pd.read_csv('AAL.csv')
roi_n=AAL.iloc[:, 0].values
roi_names=AAL.iloc[:, 1].values

weights=est2.params[1:]
weights_abs=np.absolute(weights)
weights_sorted=np.sort(weights_abs)
indices = np.argsort(weights_abs)

pvals=est2.pvalues[1:]
print pvals.shape
pv=pvals[indices[::-1]]
colors = []
for i in range(116):
    if pv[i]<0.05:
        colors.append('lawngreen')
    else:
        colors.append('deepskyblue')

hatching = []
w=weights[indices[::-1]]
for i in range(116):
    if w[i] >= 0:
        hatching.append(' ')
    else:
        hatching.append('///')
# print weights_sorted
# print indices
# print roi_names[indices[0,:]]
# fig1, ax=plt.subplots(2, 1, sharex=True, sharey=False)
# fig1.tight_layout()
# ax[0].plot(range(weights_sorted.shape[0]), pv)
# ax[0].set_ylabel('p-value')
# ax[1].bar(range(weights_sorted.shape[0]), weights_abs[indices[::-1]], color=colors)
# # Loop over the bars
# for i, thisbar in enumerate(ax[1].patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatching[i])
# ax[1].set_xticks(np.arange(0, 115, 1))
# ax[1].set_xticklabels(roi_names[indices[::-1]], rotation='vertical')
# # ax.set_title('predicting eGFR')
# ax[1].set_xlabel('AAL ROI')
# ax[1].set_ylabel('weight')
# plt.tight_layout()
# plt.show()

fig1, ax=plt.subplots()
# fig1.tight_layout()
ax.bar(range(weights_sorted.shape[0]), weights_abs[indices[::-1]], color=colors)
# Loop over the bars
for i, thisbar in enumerate(ax.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatching[i])
ax.set_xticks(np.arange(0, 115, 1))
ax.set_xticklabels(roi_names[indices[::-1]], rotation='vertical')
ax.set_xlabel('AAL ROI')
ax.set_ylabel('absolute weight')

# ax2=ax.twinx()
# ax2.plot(range(weights_sorted.shape[0]), pv, color='red')
# ax2.set_ylabel('p-value', color='red')


plt.tight_layout()
plt.show()

stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)