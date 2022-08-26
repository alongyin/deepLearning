#-*—coding:utf-8-*-
import pandas as pd
import argparse
import os
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.models import Model 
from keras.layers import Input
#全局数据参数
traindata_srcpath = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
testdata_srcpath = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


#train_data = "/root/along/deepLearning/widedeep/data/train.csv"
#test_data = "/root/along/deepLearning/widedeep/data/test.csv"

#定义样本输入格式
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

COLUMNS_DEFAULTS = [[0],[''],[0],[''],[0],[''],[''],[''],[''],[''],
                    [0],[0],[0],[''],['']]

num_examples = {
    'train': 32561,
    'validation': 16281
}

#one hot 向量化
def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

#数据准备
def get_data_download(train_data,test_data):
    """if adult data "train.csv" and "test.csv" are not in your directory,
    download them.
    """
   

    if not os.path.exists(train_data):
        print("downloading training data....")
        df_train = pd.read_csv(traindata_srcpath,names=COLUMNS, skipinitialspace=True)
        df_train.to_csv(train_data)
    else:
        df_train = pd.read_csv(train_data)


    if not os.path.exists(test_data):
        print("downloading testing data...")
        df_test = pd.read_csv(testdata_srcpath,names=COLUMNS,skipinitialspace=True)
        df_test.to_csv(test_data)
    else:
        df_test = pd.read_csv(test_data)
    return df_train,df_test


#实现：util的相关定义
def cross_columns(x_cols):
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname,x_c in zip(colnames,x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


#深度模型任务定义


#


#the structure of wide
def wide(df_train,df_test,wide_cols,x_cols,target,model_type,method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0

    #合并训练集和测试集
    df_wide = pd.concat([df_train,df_test])

    #生成交叉特征column的名称字典
    crossed_columns_d = cross_columns(x_cols)

    categorical_columns = list(df_wide.select_types(include=['object'].columns))
    wide_cols += list(crossed_columns_d.items())

    for k,v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x),axis=1)
    
    #df 特征选择，目标选择，是否训练
    df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]

    #计算需要向量化的特征列
    dummy_cols = [
        c for c in wide_cols if c in categorical_columns + list(crossed_columns_d.keys())
    ]

    #对object类，以及交叉类特征进行向量化
    df_wide = pd.get_dummies(df_wide,columns=[x for x in dummy_cols])

    train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN',axis=1)
    test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN',axis=1)
    assert all(train.columns == test.columns)

    feature_cols = [c for c in train.columns if c != target]
    X_train = train[feature_cols].values
    y_train = train[target].values.reshape(-1,1)

    X_test = test[cols].values
    y_test = test[target].values.reshape(-1,1)

    
    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    if model_type == "wide":
        activation,loss,metrics = fit_param[method]
        if metrics:
            metrics = [metrics]
        #定义wide部分的网络结构
        wide_inp = Input(shape=(X_train.shape[1],),dtype='float32',name='wide_inp')
        w = Dense(y_train.shape[1],activation=activation)(wide_inp)
        wide = Model(wide_inp,w)
        wide.complie(loss=loss,metrics = metrics,optimizer='Adam')
        wide.fit(X_train,y_train,nb_epoch=10,batch_size=64)
        results = wide.evaluate(X_test,y_test)
        print("\n",results)
    else:
        return X_train,y_train,X_test,y_test


    return 0

def deep(df_train,df_test,embedding_cols,cont_cols,target,model_type,method):
    return 0


def wide_deep(df_train,df_test,wide_cols,x_cols,embedding_cols,cont_cols,method):
    return 0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",type=str,default='logistic',help='fitting method')
    ap.add_argument("--modle_type",type=str,default='wid_deep',help='wide,deep or both')
    ap.add_argument("--train_data",type=str,default='train.csv')
    ap.add_argument("--test_data",type=str,default='test.csv')
    args = vars(ap.parse_args())
    method = args["method"]
    model_type = args["modle_type"]
    train_data = args["train_data"]
    test_data = args["test_data"]


    fit_param = dict()
    fit_param["logistic"] = ('sigmoid','binary_crossentropy','accuracy')
    fit_param['regression'] = (None,'mse',None)
    fit_param['multiclass'] = ('softmax','categorical_crossetropy','accuracy')

    df_train,df_test = get_data_download(train_data,test_data)


    #样本label生成
    df_train['income_label']=(df_train['income_bracket'].apply(lambda x: ">50K" in str(x))).astype(int)
    df_test['income_label']=(df_test['income_bracket'].apply(lambda x: ">50K" in str(x))).astype(int)

    #年龄特征离散化
    age_groups = [0,25,65,90]
    age_labels = range(len(age_groups)-1)
    df_train['age_group'] = pd.cut(df_train['age'],age_groups,labels=age_labels)
    
    df_test['isDigit'] = df_test['age'].str.isdigit()
    df_test = df_test[df_test['isDigit']==True]
    df_test['age2'] = pd.to_numeric(df_test['age'])
    #print(df_test['age2'].dtypes)
    df_test['age_group'] = pd.cut(df_test['age2'],age_groups,labels=age_labels)
    

    #columns for wide model
    wide_cols = ['workclass','education','marital_status','occupation'
    ,'relationship','race','gender','native_country','age_group']
    x_cols = (['education','occupation'],['native_country','occupation'])

    #columns for deep model
    embedding_cols = ['workclass','education','marital_status','occupation'
    ,'relationship','race','gender','native_country']
    cont_cols = ['age','capital_gain','capital_loss','hours_per_week']

    #target for logistic
    target = 'income_label'


    if model_type == "wide":
        wide(df_train,df_test,wide_cols,x_cols,target,model_type,method)
    elif model_type == "deep":
        deep(df_train,df_test,embedding_cols,cont_cols,target,model_type,method)
    else:
        wide_deep(df_train,df_test,wide_cols,x_cols,embedding_cols,cont_cols,method)
    





    print("method:",method)
    print("model_type:",model_type)
    print("train_data:",train_data)
    print("test_data",test_data)


