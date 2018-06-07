#encoding=utf-8
import pandas as pd
data = [[1,2,3],[4,5,6]]
index = ['e','f']
columns=['a','b','c']
df = pd.DataFrame(data=data, index=index, columns=columns)

#drop的测试
df=df.drop(['b','c'],axis=1)
print(df)
