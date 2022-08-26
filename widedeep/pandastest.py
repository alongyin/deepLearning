import pandas as pd
import numpy as np
df = pd.DataFrame(np.array(([1,2,3],[4,5,6])),
                  index=['mouse','rabbit'],
                  columns=['one','two','three'])
print(df)

# 过滤列
print(df.filter(items=['one','three']))
print(df.filter(['one']))
print(df.filter(regex='e$',axis=1))
print(df.filter(regex='e$',axis=0))
print(df.filter(regex='Q'))
print(df.filter(like='bb',axis=0))
print(df.filter(['one','two'],axis=1))
print(df.filter(regex='^r',axis=0).filter(like='o',axis=1))
print(df[df.one != '2'])
