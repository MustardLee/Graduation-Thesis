#!/usr/bin/env python
# coding: utf-8

# # 毕业论文代码
# #### *@author: Rebecca Li*
# 
# 首先引入packages和dataset

# In[127]:


import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


# In[4]:


bonds = pd.read_excel('/Users/rebecca/Desktop/Thesis/Data/database_v4.3.xlsx', sheet_name = 'Sheet1')


#  

# 查看数据表头

# In[3]:


bonds.head()


# 查看数据集缺失值情况

# In[78]:


bonds.isnull().sum(axis=0).sort_values(ascending=False)/float(len(bonds))


#  

# 查看数据集的维数和数据集的列名

# In[110]:


bonds.shape
bonds.columns


# In[112]:


# 从列标题中选择特征变量(features)

cols_of_feature = bonds.columns[2:] 
cols_of_feature


# In[117]:


# 目标变量分布(Frequency)可视化
fig, axs = plt.subplots(1,2,figsize=(20,12))

sns.countplot(x='Rank',data=bonds,palette="Set3",ax=axs[0])
axs[0].set_title("Frequency of each Rank")
bonds['Rank'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Target")
fig.savefig("Frequency & Percentage of Sample Target")
plt.show() 


# In[118]:


# 查看数据集中各类别的个数
bonds['Rank'].value_counts()


# In[119]:


# 在不同目标变量下，对比指标的相关性，并绘制相关性图谱
RankI = bonds.loc[bonds["Rank"] == "I"] 
RankII = bonds.loc[bonds["Rank"] == "II"] 
RankIII = bonds.loc[bonds["Rank"] == "III"]


# In[120]:


# Rank I相关性图谱
correlationRankI = RankI.loc[:, bonds.columns.difference(['Rank', 'BondCode'])].corr()

mask = np.zeros_like(correlationRankI)
indices = np.triu_indices_from(correlationRankI)
mask[indices] = True

with sns.axes_style("white"):
    f, ax1 = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(220, 8, as_cmap=True)

    ax1 =sns.heatmap(correlationRankI, vmin = -1, vmax = 1, cmap = cmap, square = False, mask = mask, linewidths = 0.1, cbar_kws={'orientation': 'vertical','ticks': [-1, -0.5, 0, 0.5, 1]})
    ax1.set_xticklabels(ax1.get_xticklabels(), size = 8)
    ax1.set_yticklabels(ax1.get_yticklabels(), size = 8)
    ax1.set_title('Rank I Correlation', size = 20)

plt.savefig('Rank I Correlation', bbox_inches='tight') 

#grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
# f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))


# In[121]:


correlationRankII = RankII.loc[:, bonds.columns.difference(['Rank', 'BondCode'])].corr()
correlationRankIII = RankIII.loc[:, bonds.columns.difference(['Rank', 'BondCode'])].corr()

mask = np.zeros_like(correlationRankII)
indices = np.triu_indices_from(correlationRankII)
mask[indices] = True
grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}

f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))

cmap = sns.diverging_palette(220, 8, as_cmap=True)


ax1 =sns.heatmap(correlationRankII, ax = ax1, vmin = -1, vmax = 1, cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)
ax1.set_xticklabels(ax1.get_xticklabels(), size = 8)
ax1.set_yticklabels(ax1.get_yticklabels(), size = 8)
ax1.set_title('RankII', size = 20)


ax2 = sns.heatmap(correlationRankIII, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False, cbar_ax = cbar_ax,cbar_kws={'orientation': 'vertical','ticks': [-1, -0.5, 0, 0.5, 1]})
ax2.set_xticklabels(ax2.get_xticklabels(), size = 8)
ax2.set_title('RankIII', size = 20)
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 12)
plt.savefig('Rank I & Rank III Feature Correlation', bbox_inches='tight') 


# ## 指标选取Feature Selection

# In[123]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

X=bonds.drop(['Rank', 'BondCode'], axis=1)  # Features
y=bonds[['Rank']] # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # 75% training and 25% test


# In[124]:


# Create Random Forest Classifier
clf=RandomForestClassifier(n_estimators = 10,random_state = 123)


# In[125]:


# Fitting Original Data
clf.fit(X_train, np.ravel(y_train)) 


#  

# 获取RF模型精确度(Accuracy)

# 1. metrics方法获取精确度

# In[133]:


#获得测试集预测值
y_pred=clf.predict(X_test)
# Model Accuracy(how often is the classifier correct)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# 2. 内置score方法获取精确度

# In[134]:


# Get the mean accuracy on the given test data and labels.
print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))


# 获取目前指标重要性排序(Identify And Select Most Important Features)

# In[135]:


importance = clf.feature_importances_
feature_names = np.array(cols_of_feature)

# for feature in zip(cols_of_feature, clf.feature_importances_):
#     print(feature)

#fe_co = plt.bar(height=importance, x=feature_names)
plt.subplots(figsize = (20,15))
plt.barh(feature_names, importance, align='center') 
plt.yticks(feature_names) 
plt.xlabel("Feature importance")
plt.ylabel("Feature")

plt.savefig('Feature Correlation Original', bbox_inches='tight') 


# In[136]:


feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importance)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# 由此可知，需剔除两个弱相关(<15%)变量if_public和if_basic

# In[138]:


#根据输出的各指标重要性排序结果，剔除重要性最弱的两个变量Q、H
cols_to_drop = ['if_basic','if_public']
X_new = X.drop(cols_to_drop, axis=1)
X_new.head()


# ## 运用新数据构建默认模型(default model RFC)

# In[139]:


#默认值查看
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.25)

clf0 = RandomForestClassifier(oob_score=True, random_state=123)
clf0.fit(X_train_new,np.ravel(y_train_new))
print(clf0.oob_score_)


# 袋外分数0.9310842303850849，较良好；<br>
# 通过调参对现有默认模型依据OOB error进行改进，其中改进过称见附录

# # 构建最终模型(Final RFC)

# In[188]:


#Final Model: CLF

clf1 = RandomForestClassifier(n_estimators = 503, oob_score=True, random_state=123, max_depth=32)
clf1.fit(X_train_new,np.ravel(y_train_new))
y_pred1=clf1.predict(X_test_new)


# 获取准确度Accuracy

# In[189]:


print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred1))
print("Accuracy on training set: {:.5f}".format(clf1.score(X_train_new, y_train_new)))
print("Accuracy on test set: {:.5f}".format(clf1.score(X_test_new, y_test_new)))


# 获取OOB Error

# In[190]:


#OOB Score indicate whether the model is overfitting
OOB_score = clf1.oob_score_

print("OOB Error: {:.5f}".format(1 - OOB_score))


# In[191]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test_new, y_pred1))
print(classification_report(y_test_new, y_pred1))


# 解释：<br>
# * precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；也就是模型预测为I/II的值中，有多少是正确的
# * recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；也就是对于原值为I/II的值，有多少预测正确了
# * f1-score:F1 = 2*精度*召回率/(精度+召回率)
# * micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6

# 新模型Feature重要性排序

# In[192]:


importance1 = clf1.feature_importances_
cols_of_feature1 = X_new.columns
feature_names1 = np.array(cols_of_feature1)

plt.subplots(figsize = (20,15))
plt.barh(feature_names1, importance1, align='center') 
plt.yticks(feature_names1) 
plt.xlabel("Feature importance for Random Forest")
plt.ylabel("Feature")

plt.savefig('Feature Correlation Final', bbox_inches='tight') 


# 可见，总资产(对数值)和收益率，是否担保及流动比率最为重要

# In[193]:


feature_importances1 = [(feature, round(importance1, 2)) for feature, importance1 in zip(feature_names1, importance1)]

# Sort the feature importances by most important first
feature_importances1 = sorted(feature_importances1, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances1];


# # 附录：调参过称

# ### 1. 对n_estimators进行调参

# In[157]:


# Range of `n_estimators` values to explore.
min_estimators = 23
max_estimators = 800

error_rate = {n : 0 for n in range(min_estimators, max_estimators + 1, 5)}


# In[158]:


clf3 = RandomForestClassifier(oob_score=True, random_state=123)

for i in range(min_estimators, max_estimators + 1, 5):
    clf3.set_params(n_estimators=i)
    clf3.fit(X_train_new,np.ravel(y_train_new))

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf3.oob_score_
    error_rate[i] = oob_error


# In[159]:


plt.plot(list(error_rate.keys()), list(error_rate.values()))

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for n', bbox_inches='tight') 


# In[160]:


#按照error_rate的value进行排序

error_sort = sorted(error_rate.items(), key = lambda kv:(kv[1], kv[0]))
error_sort[:9]
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in error_sort]


# n_estimators根据OOB取最小值503

# ### 3. 对max_features进行调参

# In[181]:


error_rate4 = {n : 0 for n in range(2, 24, 2)}


# In[182]:


clf4 = RandomForestClassifier(n_estimators = 503, oob_score=True, max_depth = 32, random_state=123)

for i in range(2, 22, 2):
    clf4.set_params(max_features = i)
    clf4.fit(X_train_new,np.ravel(y_train_new))

    oob_error = 1 - clf4.oob_score_
    error_rate4[i] = oob_error
    
    print('Counting on:', i)


# In[185]:


plt.plot(list(error_rate4.keys()), list(error_rate4.values()))

plt.xlim(2, 20)
plt.xlabel("max_features")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for features', bbox_inches='tight') 


# In[187]:


error_sort4 = sorted(error_rate4.items(), key = lambda kv:(kv[1], kv[0]))
error_sort4[:19]


# max_features取合理区间$[4,6]$OOB最小时值为4,近似sqrt(20),因此最终模型取默认值

# ### 2.对决策树最大深度max_depth进行调参

# >若等于None,表示决策树在构建最优模型的时候不会限制子树的深度。如果模型样本量多，特征也多的情况下，推荐限制最大深度；若样本量少或者特征少，则不限制最大深度。

# In[161]:


error_rate5 = {n : 0 for n in range(4,40,2)}


# In[162]:



clf5 = RandomForestClassifier(n_estimators = 503, oob_score=True, random_state=123)

for i in range(4,40,2):
    clf5.set_params(max_depth = i)
    clf5.fit(X_train_new,np.ravel(y_train_new))

    oob_error = 1 - clf5.oob_score_
    error_rate5[i] = oob_error


# In[163]:


plt.plot(list(error_rate5.keys()), list(error_rate5.values()))

plt.xlim(4,40,2)
plt.xlabel("max_depth")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for depth', bbox_inches='tight') 


# In[164]:


error_sort5 = sorted(error_rate5.items(), key = lambda kv:(kv[1], kv[0]))
error_sort5[:40]


# max_depth取OOB最小时值32

# In[ ]:




