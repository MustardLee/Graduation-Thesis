# 毕业论文代码
#### *@author: Rebecca Li*

首先引入packages和dataset


```python
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
```


```python
bonds = pd.read_excel('/Users/rebecca/Desktop/Thesis/Data/database_v4.3.xlsx', sheet_name = 'Sheet1')
```

 

查看数据表头


```python
bonds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BondCode</th>
      <th>Rank</th>
      <th>issueamount</th>
      <th>term/year</th>
      <th>yield%</th>
      <th>Public_Revenue</th>
      <th>fa_yoy_or</th>
      <th>ROE</th>
      <th>EBITDA2D</th>
      <th>Current</th>
      <th>...</th>
      <th>Salescash2OR</th>
      <th>OM</th>
      <th>EG</th>
      <th>CAreturn</th>
      <th>Invreturn</th>
      <th>LNProvince_GDP</th>
      <th>is_gurarantee</th>
      <th>LNTotal_Asset</th>
      <th>if_public</th>
      <th>if_basic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>078080.IB</td>
      <td>I</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>6.08</td>
      <td>428.0265</td>
      <td>-8.8139</td>
      <td>2.9081</td>
      <td>0.038838</td>
      <td>2.4302</td>
      <td>...</td>
      <td>98.5065</td>
      <td>-70.1820</td>
      <td>-56.874827</td>
      <td>0.0297</td>
      <td>0.0309</td>
      <td>8.718091</td>
      <td>False</td>
      <td>4.202351</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122932.SH</td>
      <td>I</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>724.6197</td>
      <td>21.7038</td>
      <td>2.1190</td>
      <td>0.042112</td>
      <td>6.8488</td>
      <td>...</td>
      <td>122.0677</td>
      <td>-120.1565</td>
      <td>-46.002507</td>
      <td>0.0146</td>
      <td>0.0158</td>
      <td>9.088360</td>
      <td>True</td>
      <td>4.774094</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0980167.IB</td>
      <td>I</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>724.6197</td>
      <td>21.7038</td>
      <td>2.1190</td>
      <td>0.042112</td>
      <td>6.8488</td>
      <td>...</td>
      <td>122.0677</td>
      <td>-120.1565</td>
      <td>-46.002507</td>
      <td>0.0146</td>
      <td>0.0158</td>
      <td>9.088360</td>
      <td>True</td>
      <td>4.774094</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124307.SH</td>
      <td>I</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.00</td>
      <td>1792.7192</td>
      <td>200.5891</td>
      <td>17.7061</td>
      <td>0.975419</td>
      <td>4.1721</td>
      <td>...</td>
      <td>44.4314</td>
      <td>96.1972</td>
      <td>272.069235</td>
      <td>0.2838</td>
      <td>0.1689</td>
      <td>9.753365</td>
      <td>False</td>
      <td>4.325516</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1380222.IB</td>
      <td>I</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.00</td>
      <td>1792.7192</td>
      <td>200.5891</td>
      <td>17.7061</td>
      <td>0.975419</td>
      <td>4.1721</td>
      <td>...</td>
      <td>44.4314</td>
      <td>96.1972</td>
      <td>272.069235</td>
      <td>0.2838</td>
      <td>0.1689</td>
      <td>9.753365</td>
      <td>False</td>
      <td>4.325516</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



查看数据集缺失值情况


```python
bonds.isnull().sum(axis=0).sort_values(ascending=False)/float(len(bonds))
```




    if_basic          0.0
    if_public         0.0
    Rank              0.0
    issueamount       0.0
    term/year         0.0
    yield%            0.0
    Public_Revenue    0.0
    fa_yoy_or         0.0
    ROE               0.0
    EBITDA2D          0.0
    Current           0.0
    D2A               0.0
    CF2D              0.0
    CF2I              0.0
    Earning2A         0.0
    Salescash2OR      0.0
    OM                0.0
    EG                0.0
    CAreturn          0.0
    Invreturn         0.0
    LNProvince_GDP    0.0
    is_gurarantee     0.0
    LNTotal_Asset     0.0
    BondCode          0.0
    dtype: float64



 

查看数据集的维数和数据集的列名


```python
bonds.shape
bonds.columns
```




    Index(['BondCode', 'Rank', 'issueamount', 'term/year', 'yield%',
           'Public_Revenue', 'fa_yoy_or', 'ROE', 'EBITDA2D', 'Current', 'D2A',
           'CF2D', 'CF2I', 'Earning2A', 'Salescash2OR', 'OM', 'EG', 'CAreturn',
           'Invreturn', 'LNProvince_GDP', 'is_gurarantee', 'LNTotal_Asset',
           'if_public', 'if_basic'],
          dtype='object')




```python
# 从列标题中选择特征变量(features)

cols_of_feature = bonds.columns[2:] 
cols_of_feature
```




    Index(['issueamount', 'term/year', 'yield%', 'Public_Revenue', 'fa_yoy_or',
           'ROE', 'EBITDA2D', 'Current', 'D2A', 'CF2D', 'CF2I', 'Earning2A',
           'Salescash2OR', 'OM', 'EG', 'CAreturn', 'Invreturn', 'LNProvince_GDP',
           'is_gurarantee', 'LNTotal_Asset', 'if_public', 'if_basic'],
          dtype='object')




```python
# 目标变量分布(Frequency)可视化
fig, axs = plt.subplots(1,2,figsize=(20,12))

sns.countplot(x='Rank',data=bonds,palette="Set3",ax=axs[0])
axs[0].set_title("Frequency of each Rank")
bonds['Rank'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Target")
fig.savefig("Frequency & Percentage of Sample Target")
plt.show() 
```


![png](output_12_0.png)



```python
# 查看数据集中各类别的个数
bonds['Rank'].value_counts()
```




    I      15035
    II      8979
    III       16
    Name: Rank, dtype: int64




```python
# 在不同目标变量下，对比指标的相关性，并绘制相关性图谱
RankI = bonds.loc[bonds["Rank"] == "I"] 
RankII = bonds.loc[bonds["Rank"] == "II"] 
RankIII = bonds.loc[bonds["Rank"] == "III"]
```


```python
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
```


![png](output_15_0.png)



```python
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
```


![png](output_16_0.png)


## 指标选取Feature Selection


```python
# Import train_test_split function
from sklearn.model_selection import train_test_split

X=bonds.drop(['Rank', 'BondCode'], axis=1)  # Features
y=bonds[['Rank']] # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # 75% training and 25% test
```


```python
# Create Random Forest Classifier
clf=RandomForestClassifier(n_estimators = 10,random_state = 123)
```


```python
# Fitting Original Data
clf.fit(X_train, np.ravel(y_train)) 
```




    RandomForestClassifier(n_estimators=10, random_state=123)



 

获取RF模型精确度(Accuracy)

1. metrics方法获取精确度


```python
#获得测试集预测值
y_pred=clf.predict(X_test)
# Model Accuracy(how often is the classifier correct)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

    Accuracy: 0.9181091877496671


2. 内置score方法获取精确度


```python
# Get the mean accuracy on the given test data and labels.
print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
```

    Accuracy on training set: 0.996
    Accuracy on test set: 0.918


获取目前指标重要性排序(Identify And Select Most Important Features)


```python
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
```


![png](output_28_0.png)



```python
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importance)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
```

    Variable: LNTotal_Asset        Importance: 0.22
    Variable: yield%               Importance: 0.11
    Variable: issueamount          Importance: 0.05
    Variable: Current              Importance: 0.05
    Variable: is_gurarantee        Importance: 0.05
    Variable: Public_Revenue       Importance: 0.04
    Variable: Earning2A            Importance: 0.04
    Variable: Salescash2OR         Importance: 0.04
    Variable: OM                   Importance: 0.04
    Variable: CAreturn             Importance: 0.04
    Variable: Invreturn            Importance: 0.04
    Variable: term/year            Importance: 0.03
    Variable: ROE                  Importance: 0.03
    Variable: EBITDA2D             Importance: 0.03
    Variable: D2A                  Importance: 0.03
    Variable: CF2D                 Importance: 0.03
    Variable: CF2I                 Importance: 0.03
    Variable: EG                   Importance: 0.03
    Variable: LNProvince_GDP       Importance: 0.03
    Variable: fa_yoy_or            Importance: 0.02
    Variable: if_public            Importance: 0.01
    Variable: if_basic             Importance: 0.0


由此可知，需剔除两个弱相关(<15%)变量if_public和if_basic


```python
#根据输出的各指标重要性排序结果，剔除重要性最弱的两个变量Q、H
cols_to_drop = ['if_basic','if_public']
X_new = X.drop(cols_to_drop, axis=1)
X_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>issueamount</th>
      <th>term/year</th>
      <th>yield%</th>
      <th>Public_Revenue</th>
      <th>fa_yoy_or</th>
      <th>ROE</th>
      <th>EBITDA2D</th>
      <th>Current</th>
      <th>D2A</th>
      <th>CF2D</th>
      <th>CF2I</th>
      <th>Earning2A</th>
      <th>Salescash2OR</th>
      <th>OM</th>
      <th>EG</th>
      <th>CAreturn</th>
      <th>Invreturn</th>
      <th>LNProvince_GDP</th>
      <th>is_gurarantee</th>
      <th>LNTotal_Asset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>10.0</td>
      <td>6.08</td>
      <td>428.0265</td>
      <td>-8.8139</td>
      <td>2.9081</td>
      <td>0.038838</td>
      <td>2.4302</td>
      <td>53.6158</td>
      <td>0.0459</td>
      <td>11.397205</td>
      <td>7.728968</td>
      <td>98.5065</td>
      <td>-70.1820</td>
      <td>-56.874827</td>
      <td>0.0297</td>
      <td>0.0309</td>
      <td>8.718091</td>
      <td>False</td>
      <td>4.202351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>724.6197</td>
      <td>21.7038</td>
      <td>2.1190</td>
      <td>0.042112</td>
      <td>6.8488</td>
      <td>53.4305</td>
      <td>0.0222</td>
      <td>1.403246</td>
      <td>7.144044</td>
      <td>122.0677</td>
      <td>-120.1565</td>
      <td>-46.002507</td>
      <td>0.0146</td>
      <td>0.0158</td>
      <td>9.088360</td>
      <td>True</td>
      <td>4.774094</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.0</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>724.6197</td>
      <td>21.7038</td>
      <td>2.1190</td>
      <td>0.042112</td>
      <td>6.8488</td>
      <td>53.4305</td>
      <td>0.0222</td>
      <td>1.403246</td>
      <td>7.144044</td>
      <td>122.0677</td>
      <td>-120.1565</td>
      <td>-46.002507</td>
      <td>0.0146</td>
      <td>0.0158</td>
      <td>9.088360</td>
      <td>True</td>
      <td>4.774094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.00</td>
      <td>1792.7192</td>
      <td>200.5891</td>
      <td>17.7061</td>
      <td>0.975419</td>
      <td>4.1721</td>
      <td>39.9249</td>
      <td>0.1545</td>
      <td>137.334566</td>
      <td>20.004795</td>
      <td>44.4314</td>
      <td>96.1972</td>
      <td>272.069235</td>
      <td>0.2838</td>
      <td>0.1689</td>
      <td>9.753365</td>
      <td>False</td>
      <td>4.325516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.00</td>
      <td>1792.7192</td>
      <td>200.5891</td>
      <td>17.7061</td>
      <td>0.975419</td>
      <td>4.1721</td>
      <td>39.9249</td>
      <td>0.1545</td>
      <td>137.334566</td>
      <td>20.004795</td>
      <td>44.4314</td>
      <td>96.1972</td>
      <td>272.069235</td>
      <td>0.2838</td>
      <td>0.1689</td>
      <td>9.753365</td>
      <td>False</td>
      <td>4.325516</td>
    </tr>
  </tbody>
</table>
</div>



## 运用新数据构建默认模型(default model RFC)


```python
#默认值查看
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.25)

clf0 = RandomForestClassifier(oob_score=True, random_state=123)
clf0.fit(X_train_new,np.ravel(y_train_new))
print(clf0.oob_score_)
```

    0.9310842303850849


袋外分数0.9310842303850849，较良好；<br>
通过调参对现有默认模型依据OOB error进行改进，其中改进过称见附录

# 构建最终模型(Final RFC)


```python
#Final Model: CLF

clf1 = RandomForestClassifier(n_estimators = 503, oob_score=True, random_state=123, max_depth=32)
clf1.fit(X_train_new,np.ravel(y_train_new))
y_pred1=clf1.predict(X_test_new)
```

获取准确度Accuracy


```python
print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred1))
print("Accuracy on training set: {:.5f}".format(clf1.score(X_train_new, y_train_new)))
print("Accuracy on test set: {:.5f}".format(clf1.score(X_test_new, y_test_new)))
```

    Accuracy: 0.938249001331558
    Accuracy on training set: 0.99989
    Accuracy on test set: 0.93825


获取OOB Error


```python
#OOB Score indicate whether the model is overfitting
OOB_score = clf1.oob_score_

print("OOB Error: {:.5f}".format(1 - OOB_score))
```

    OOB Error: 0.06331



```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test_new, y_pred1))
print(classification_report(y_test_new, y_pred1))
```

    [[3546  177    0]
     [ 190 2090    0]
     [   1    3    1]]
                  precision    recall  f1-score   support
    
               I       0.95      0.95      0.95      3723
              II       0.92      0.92      0.92      2280
             III       1.00      0.20      0.33         5
    
        accuracy                           0.94      6008
       macro avg       0.96      0.69      0.73      6008
    weighted avg       0.94      0.94      0.94      6008



解释：<br>
* precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；也就是模型预测为I/II的值中，有多少是正确的
* recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；也就是对于原值为I/II的值，有多少预测正确了
* f1-score:F1 = 2*精度*召回率/(精度+召回率)
* micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6

新模型Feature重要性排序


```python
importance1 = clf1.feature_importances_
cols_of_feature1 = X_new.columns
feature_names1 = np.array(cols_of_feature1)

plt.subplots(figsize = (20,15))
plt.barh(feature_names1, importance1, align='center') 
plt.yticks(feature_names1) 
plt.xlabel("Feature importance for Random Forest")
plt.ylabel("Feature")

plt.savefig('Feature Correlation Final', bbox_inches='tight') 
```


![png](output_44_0.png)


可见，总资产(对数值)和收益率，是否担保及流动比率最为重要


```python
feature_importances1 = [(feature, round(importance1, 2)) for feature, importance1 in zip(feature_names1, importance1)]

# Sort the feature importances by most important first
feature_importances1 = sorted(feature_importances1, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances1];
```

    Variable: LNTotal_Asset        Importance: 0.23
    Variable: yield%               Importance: 0.11
    Variable: is_gurarantee        Importance: 0.06
    Variable: issueamount          Importance: 0.05
    Variable: Current              Importance: 0.05
    Variable: D2A                  Importance: 0.04
    Variable: Earning2A            Importance: 0.04
    Variable: CAreturn             Importance: 0.04
    Variable: Invreturn            Importance: 0.04
    Variable: term/year            Importance: 0.03
    Variable: Public_Revenue       Importance: 0.03
    Variable: fa_yoy_or            Importance: 0.03
    Variable: ROE                  Importance: 0.03
    Variable: EBITDA2D             Importance: 0.03
    Variable: CF2D                 Importance: 0.03
    Variable: CF2I                 Importance: 0.03
    Variable: Salescash2OR         Importance: 0.03
    Variable: OM                   Importance: 0.03
    Variable: EG                   Importance: 0.03
    Variable: LNProvince_GDP       Importance: 0.03


# 附录：调参过称

### 1. 对n_estimators进行调参


```python
# Range of `n_estimators` values to explore.
min_estimators = 23
max_estimators = 800

error_rate = {n : 0 for n in range(min_estimators, max_estimators + 1, 5)}
```


```python
clf3 = RandomForestClassifier(oob_score=True, random_state=123)

for i in range(min_estimators, max_estimators + 1, 5):
    clf3.set_params(n_estimators=i)
    clf3.fit(X_train_new,np.ravel(y_train_new))

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf3.oob_score_
    error_rate[i] = oob_error
```


```python
plt.plot(list(error_rate.keys()), list(error_rate.values()))

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for n', bbox_inches='tight') 
```


![png](output_51_0.png)



    <Figure size 432x288 with 0 Axes>



```python
#按照error_rate的value进行排序

error_sort = sorted(error_rate.items(), key = lambda kv:(kv[1], kv[0]))
error_sort[:9]
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in error_sort]
```




    [(503, 0.06336699589390749),
     (498, 0.06347797136832756),
     (393, 0.06353345910553765),
     (438, 0.06353345910553765),
     (388, 0.06358894684274774),
     (408, 0.06358894684274774),
     (403, 0.06364443457995783),
     (413, 0.06364443457995783),
     (423, 0.06364443457995783)]



n_estimators根据OOB取最小值503

### 3. 对max_features进行调参


```python
error_rate4 = {n : 0 for n in range(2, 24, 2)}
```


```python
clf4 = RandomForestClassifier(n_estimators = 503, oob_score=True, max_depth = 32, random_state=123)

for i in range(2, 22, 2):
    clf4.set_params(max_features = i)
    clf4.fit(X_train_new,np.ravel(y_train_new))

    oob_error = 1 - clf4.oob_score_
    error_rate4[i] = oob_error
    
    print('Counting on:', i)
```

    Counting on: 2
    Counting on: 4
    Counting on: 6
    Counting on: 8
    Counting on: 10
    Counting on: 12
    Counting on: 14
    Counting on: 16
    Counting on: 18
    Counting on: 20



```python
plt.plot(list(error_rate4.keys()), list(error_rate4.values()))

plt.xlim(2, 20)
plt.xlabel("max_features")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for features', bbox_inches='tight') 
```


![png](output_57_0.png)



    <Figure size 432x288 with 0 Axes>



```python
error_sort4 = sorted(error_rate4.items(), key = lambda kv:(kv[1], kv[0]))
error_sort4[:19]
```




    [(22, 0),
     (2, 0.061757851514815276),
     (4, 0.0633115081566974),
     (6, 0.06508711574741977),
     (10, 0.06591943180557092),
     (8, 0.0660304072799911),
     (14, 0.06625235822883146),
     (12, 0.06630784596604156),
     (18, 0.0665852846520919),
     (16, 0.06680723560093216),
     (20, 0.06780601487071358)]



max_features取合理区间$[4,6]$OOB最小时值为4,近似sqrt(20),因此最终模型取默认值

### 2.对决策树最大深度max_depth进行调参

>若等于None,表示决策树在构建最优模型的时候不会限制子树的深度。如果模型样本量多，特征也多的情况下，推荐限制最大深度；若样本量少或者特征少，则不限制最大深度。


```python
error_rate5 = {n : 0 for n in range(4,40,2)}
```


```python

clf5 = RandomForestClassifier(n_estimators = 503, oob_score=True, random_state=123)

for i in range(4,40,2):
    clf5.set_params(max_depth = i)
    clf5.fit(X_train_new,np.ravel(y_train_new))

    oob_error = 1 - clf5.oob_score_
    error_rate5[i] = oob_error

```


```python
plt.plot(list(error_rate5.keys()), list(error_rate5.values()))

plt.xlim(4,40,2)
plt.xlabel("max_depth")
plt.ylabel("OOB error rate")
plt.show()
plt.savefig('OOB Error Rate for depth', bbox_inches='tight') 
```


![png](output_64_0.png)



    <Figure size 432x288 with 0 Axes>



```python
error_sort5 = sorted(error_rate5.items(), key = lambda kv:(kv[1], kv[0]))
error_sort5[:40]
```




    [(32, 0.0633115081566974),
     (34, 0.0633115081566974),
     (36, 0.06336699589390749),
     (38, 0.06336699589390749),
     (30, 0.06369992231716792),
     (28, 0.06419931195205864),
     (20, 0.06425479968926873),
     (26, 0.06431028742647871),
     (22, 0.06447675063810898),
     (24, 0.06464321384973926),
     (18, 0.06569748085673066),
     (16, 0.0699145488846965),
     (14, 0.07690600377316614),
     (12, 0.08872489179891241),
     (10, 0.10858950172011983),
     (8, 0.12967484185994893),
     (6, 0.14942847630673617),
     (4, 0.17045832870935518)]



max_depth取OOB最小时值32


```python

```
