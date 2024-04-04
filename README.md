## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![Screenshot 2024-04-04 111626](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/63fd3945-c97f-4756-b34a-6869beca1fe5)

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![Screenshot 2024-04-04 111631](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/f091e54e-98a7-48a6-a176-7efe895eb63d)

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![Screenshot 2024-04-04 111636](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/330bf243-e629-42fc-b489-2148701f5f26)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2024-04-04 111647](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/626843d4-f475-4e4c-a6d4-0244f0a72f6b)

```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```

![Screenshot 2024-04-04 111659](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/4aa98ca9-b120-4683-bcbd-4add830db08a)

```python
df2=pd.concat([df2,enc],axis=1)
dfs=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2024-04-04 111709](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/3bf9819c-c032-449f-b0d7-e00ea17d8647)

```python
pd.get_dummies(df2,columns=["nom_0"])
```


```python
pip install --upgrade category_encoders
```


```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```


```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy
dfb
```


```python
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```


```python
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")
df
```


```python
df.skew()
```

![Screenshot 2024-04-04 111802](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/fad442db-3ad9-405e-8899-933333abb4e8)

```python
np.log(df["Highly Positive Skew"])
```

![Screenshot 2024-04-04 111806](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/147133e7-1fc1-4cd0-b21f-d7fd76cde1c5)

```python
np.reciprocal(df["Moderate Positive Skew"])
```

![Screenshot 2024-04-04 111810](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/db1afc1e-4559-4888-b844-3e1db8d798a2)

```python
np.sqrt(df["Highly Positive Skew"])
```

![Screenshot 2024-04-04 111815](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/24cee62f-9288-4f88-abc7-1f14b572c155)

```python
np.square(df["Highly Positive Skew"])
```

![Screenshot 2024-04-04 111818](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/84df4534-e1bf-4a08-8ee1-a0d09441725f)

```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![Screenshot 2024-04-04 111825](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/0ad33d87-bb11-4a09-bfcb-8c14cb77384b)

```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

![Screenshot 2024-04-04 111831](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/acd0b6c0-b423-4cee-9f15-9c0f20d0d4aa)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![Screenshot 2024-04-04 111834](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/f67f84dc-a47c-42b6-a2e4-84ba92bdc10d)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![Screenshot 2024-04-04 111847](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/a543471d-bbff-45cb-a489-4753d9b1f62b)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-04-04 111853](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/b90f72c5-f674-47c1-a9fd-5cd68ce32581)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![Screenshot 2024-04-04 111859](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/64b4aac4-a2a5-4957-a5a3-ae5465d55776)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-04-04 111904](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/82a7fdde-7402-4488-9f6b-54955bdb4c31)


```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-04-04 111909](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/60d6c48c-a27e-490d-9fe6-daefa54d6e28)


```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![Screenshot 2024-04-04 111914](https://github.com/chandrumathiyazhagan/EXNO-3-DS/assets/119393023/aba98554-8b84-4764-9b47-5782eb8ba0b5)



## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
