# Date: 08/02/2017
# Author: Behrouz Saghafi
import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler
import numpy as np
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt


df = pd.read_csv('AADHSMind_JJ_Data20151124_RelevantColumns_v2ColorCoded_CACandCRPadded_JUST_THE_SUBJECTS_WITH_GOOD_ImagesDataColsEU_EV_FK_FP_FW.csv')
#10 disease measures and 6 confounders:
df=df.dropna(subset=['duration','hemoglobin_a1c','ckd_gfr','acr','CAC','CRP','bun','potassium_serum','protein_total_serum','microalbumin_urine','age','bmi','education','sex','hypertension','smoking'])

df['sex']=df['sex'].astype('category')
df['sex']=df['sex'].cat.codes

arr=df[['id','date11','duration','hemoglobin_a1c','ckd_gfr','acr','CAC','CRP','bun','potassium_serum','protein_total_serum','microalbumin_urine','age','bmi','education','sex','hypertension','smoking']]

print arr.shape
# print arr.iloc[1,0]
# #10 disease measures and 6 confounders
# dur = df['duration'].values
# a1c = df['hemoglobin_a1c'].values
# gfr = df['ckd_gfr'].values
# acr = df['acr'].values
# cac = df['CAC'].values
# crp = df['CRP'].values
# bun = df['bun'].values
# kse = df['potassium_serum'].values
# pts = df['protein_total_serum'].values
# alb = df['microalbumin_urine'].values
#
# #6 confounders:
# age = df['age'].values
# bmi = df['bmi'].values
# edu = df['education'].values
# sex = df['sex'].values
# hyp = df['hypertension'].values
# smk = df['smoking'].values

MR_ID=[]
for i in range(arr.shape[0]):
    ID=arr.iloc[i,0]
    if ID<100000:
        name = 'AADHS' + '0' + str(int(arr.iloc[i, 0])) + '_' + str(int(arr.iloc[i, 1]))
    else:
        name = 'AADHS' + str(int(arr.iloc[i, 0])) + '_' + str(int(arr.iloc[i, 1]))
    # print name
    MR_ID.append(name)

MR_ID = np.reshape(MR_ID, (arr.shape[0], 1))
arr = np.concatenate((arr, MR_ID), axis=1)
print arr.shape
dfa = pd.DataFrame(data=arr, columns=['id','date11','duration','hemoglobin_a1c','ckd_gfr','acr','CAC','CRP','bun','potassium_serum','protein_total_serum','microalbumin_urine','age','bmi','education','sex','hypertension','smoking','MR_ID'])

# CBF csv:
dfb = pd.read_csv('ANSIR_CBF_VBM8_AALdata_GM.csv')
# dfa = pd.DataFrame(data=arr, columns=['id', 'data', 'gfr', 'MR_ID'])
dfb.fillna(dfb.median(),inplace=True)
# C = np.array(dfb)
# C=Imputer(missing_values='NaN', strategy='median', axis=1, verbose=0, copy=True)
# C = C.fit_transform(dfb.iloc[:, 5:].values)
# C=np.array(C)
# C=pd.DataFrame(data=C)
combine = pd.merge(dfa, dfb, how='inner', on='MR_ID')
print combine.shape
combine_sorted = combine.sort(['Caudate_R'])
combine_sorted.to_csv('combine_sorted.csv')
# y=combine_sorted['Cerebelum_3_R'].values
# id=combine_sorted['id'].values
# print y
# print id
N=combine.shape[0]
print N

#
N2=int(math.ceil(0.3*N))
print N2
# 2:18=all
L=combine_sorted.iloc[0:N2,2:18].values
H=combine_sorted.iloc[-N2:,2:18].values
yl=np.zeros(N2)
yh=np.ones(N2)

X=np.concatenate((L,H), axis=0)
y=np.concatenate((yl,yh))

# ntrain=72
# ntest=20
from sklearn.model_selection import StratifiedKFold, KFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
for train_index, test_index in kfold.split(X, y):
    X_train_regional=X[train_index]
    y_train_regional=y[train_index]
    X_test_regional=X[test_index]
    y_test_regional=y[test_index]
    break

# print train_index
# print test_index
# X_train_regional = np.concatenate((L[0:ntrain/2,:],H[0:ntrain/2,:]), axis=0)
# y_train_regional=np.concatenate((yl[0:ntrain/2],yh[0:ntrain/2]),axis=0)
#
# X_test_regional = np.concatenate((L[0:ntest/2,:],H[0:ntest/2,:]), axis=0)
# y_test_regional = np.concatenate((yl[0:ntest/2],yh[0:ntest/2]),axis=0)
#
# print X_train_regional.shape
# print y_train_regional.shape
# print X_test_regional.shape
# print y_test_regional.shape
#
np.save('X_train_regional',X_train_regional)
np.save('y_train_regional',y_train_regional)
np.save('X_test_regional',X_test_regional)
np.save('y_test_regional',y_test_regional)
#
# #### statistical tests ####:
# X=combine.iloc[:,2:18].values
# X=MinMaxScaler().fit_transform(X)
# X = sm.add_constant(X)
# AAL=pd.read_excel('AAL.xlsx')
# roi_n=AAL.iloc[:, 0].values
# roi_names=AAL.iloc[:, 1].values
# counter=0
# r2=[]
# fpval=[]
# features=['duration','hemoglobin_a1c','ckd_gfr','acr','CAC','CRP','bun','potassium_serum','protein_total_serum','microalbumin_urine','age','bmi','education','sex','hypertension','smoking']
# AAL=pd.read_excel('AAL.xlsx')
# roi_n=AAL.iloc[:, 0].values
# roi_names=AAL.iloc[:, 1].values
#
# feat=pd.read_excel('features_abb.xlsx')
# feat_n=feat.iloc[:, 0].values
# feat_names=feat.iloc[:, 1].values
#
# for i in range(116):
#     y=combine.iloc[:,23+i].values
#     # print('X_shape=',X.shape)
#     # print ('y shape=',y.shape)
#     est = sm.OLS(y, X)
#     est2 = est.fit()
#     r2.append(est2.rsquared)
#     fpval.append(est2.f_pvalue)
#     # print(est2.summary())
#     # print est2.pvalues[est2.pvalues<0.01]
#     if est2.f_pvalue<(0.01/116) and est2.rsquared>0.01:
#         print roi_names[i]
#         if roi_names[i]=='Caudate_R':
#
#             weights = est2.params[1:]
#             weights_abs = np.absolute(weights)
#             weights_sorted = np.sort(weights_abs)
#             indices = np.argsort(weights_abs)
#             pvals = est2.pvalues[1:]
#             print indices
#             print weights_abs[indices]
#             print pvals
#             print(est2.summary())
#             pv = pvals[indices[::-1]]
#             # colors = []
#             # for j in range(16):
#             #     if pv[j] < 0.01:
#             #         colors.append('lawngreen')
#             #     else:
#             #         colors.append('deepskyblue')
#
#             hatching = []
#             w = weights[indices[::-1]]
#             for j in range(16):
#                 if w[j] >= 0:
#                     hatching.append(' ')
#                 else:
#                     hatching.append('///')
#
#
#             fig1, ax = plt.subplots()
#             # fig1.tight_layout()
#             ax.bar(range(weights_sorted.shape[0]), weights_abs[indices[::-1]], color='deepskyblue')
#             # Loop over the bars
#             for j, thisbar in enumerate(ax.patches):
#                 # Set a different hatch for each bar
#                 thisbar.set_hatch(hatching[j])
#             ax.set_xticks(np.arange(0, 16, 1))
#             ax.set_xticklabels(feat_names[indices[::-1]], rotation='vertical')
#             ax.set_xlabel('clinical parameters (features)')
#             ax.set_ylabel('absolute weight')
#
#             # ax2 = ax.twinx()
#             # ax2.plot(range(weights_sorted.shape[0]), pv, color='red')
#             # ax2.set_ylabel('p-value', color='red')
#
#             plt.tight_layout()
#             plt.show()
#
#         print ('f_pvalue=',est2.f_pvalue)
#         print ('R2=',est2.rsquared)
#         counter=counter+1
#         print('\n')
# print counter
# fpval=np.array(fpval)
# #
# # #
# # Plot figure:
# r2_sorted=np.sort(r2)
# indices = np.argsort(r2)
# # indices=np.reshape(indices,116)
# # print indices.dtype
# # print indices.shape
#
# indices2 = np.argsort(fpval)
#
# print indices[::-1]
# print indices2[::-1]
# print r2[74]
# print fpval[74]
# print r2[71]
# print fpval[71]
# pv=fpval[indices[::-1]]
# colors = []
# print pv
# print r2_sorted
# for i in range(116):
#     if pv[i]<(0.01/116):
#         colors.append('lawngreen')
#     else:
#         colors.append('deepskyblue')
#
# hatching = []
# w=r2_sorted[::-1]
# for i in range(116):
#     if w[i] >= 0:
#         hatching.append(' ')
#     else:
#         hatching.append('///')
#
# fig2, ax=plt.subplots()
# # fig1.tight_layout()
# ax.bar(range(r2_sorted.shape[0]), r2_sorted[::-1], color=colors)
# # Loop over the bars
# for i, thisbar in enumerate(ax.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatching[i])
# ax.set_xticks(np.arange(0, 116, 1))
# ax.set_xticklabels(roi_names[indices[::-1]], rotation='vertical')
# ax.set_xlabel('AAL ROI')
# ax.set_ylabel('$R^2$')
#
# ax2=ax.twinx()
# ax2.plot(range(pv.shape[0]), pv, color='red')
# ax2.axhline(0.01/116, color='g')
# ax2.set_ylabel('$Prob_{F-statistic}$', color='red')
#
# plt.tight_layout()
# plt.show()
