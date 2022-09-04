#-*—coding:utf-8-*-
import pandas as pd
import argparse
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from keras.layers import Dense
from keras.models import Model 
from keras.layers import Input,Embedding,Reshape,concatenate,Flatten,Lambda,Dropout
from keras.regularizers import l2, l1_l2
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


#将输入进行embedding化
def embedding_input(name,n_in,n_out,reg):
    inp = Input(shape=(1,),dtype='int64',name=name)
    return inp, Embedding(n_in,n_out,input_length=1,embeddings_regularizer=l2(reg))(inp)

#将输入值进行连续化处理
def continous_input(name):
    inp = Input(shape=(1,),dtype='float32',name=name)
    return inp,Reshape((1,1))(inp)

#value值转化为idx
def val2idx(df,cols):
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()
    
    val_to_idx = dict()
    for k,v in val_types.items():
        val_to_idx[k] = {o: i for i,o in enumerate(val_types[k])}

    for k,v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])
    
    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()
    
    return df,unique_vals


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

    categorical_columns = list(df_wide.select_dtypes(include=['object']).columns)
    wide_cols += list(crossed_columns_d.keys())

    for k,v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x),axis=1)
    
    #df 特征选择，目标选择，是否训练
    print(wide_cols + [target] + ['IS_TRAIN'])
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

    X_test = test[feature_cols].values
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
        wide.compile(loss=loss,metrics = metrics,optimizer='Adam')
        wide.fit(X_train,y_train,nb_epoch=10,batch_size=64)
        results = wide.evaluate(X_test,y_test)
        print("\n",results)
    else:
        return X_train,y_train,X_test,y_test


    return 0

def deep(df_train,df_test,embedding_cols,cont_cols,target,model_type,method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train,df_test])
    deep_cols = embedding_cols + cont_cols
    df_deep = df_deep[deep_cols + [target,'IS_TRAIN']]

    scaler = StandardScaler()
    df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_deep[cont_cols]),columns = cont_cols)
    df_deep,unique_vals = val2idx(df_deep,embedding_cols)

    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN',axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN',axis=1)

    embeddings_tensors = []
    n_factors = 8
    reg = 1e-3
    
    for ec in embedding_cols:
        layer_name = ec + "_inp"
        t_inp,t_build = embedding_input(layer_name,unique_vals[ec],n_factors,reg)
        embeddings_tensors.append((t_inp,t_build))
        del(t_inp,t_build)
    
    continuous_tensors = []

    for cc in cont_cols:
        layer_name = cc + "_in"
        t_inp,t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp,t_build))
        del(t_inp,t_build)
    
    X_train = [train[c] for c in deep_cols]
    y_train = np.array(train[target].values).reshape(-1,1)
    X_test = [test[c] for c in deep_cols]
    y_test = np.array(test[target].values).reshape(-1,1)

    if method == 'muticlass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    inp_layer  = [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]

    inp_embed = [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]


    
    if model_type == "deep":
        activation,loss,metrics = fit_param[method]
        if metrics:
            metrics = [metrics]
        
        d = concatenate(inp_embed)
        d = Flatten()(d)
        d = Dense(100,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(d)
        d = Dropout(0.5)(d) # Dropout don't seem to help in this model
        d = Dense(50,activation='relu')(d)
        d = Dropout(0.5)(d)
        d = Dense(y_train.shape[1],activation=activation)(d)
        deep = Model(inp_layer,d)
        deep.compile(loss=loss,metics=metics,optimizer='Adam')
        deep.fit(X_train,y_train,batch_size=64,nb_epoch=10)
        results = deep.evaluate(X_test,y_test)
        print("\n",results)

    else:
        return X_train,y_train,X_test,y_test,inp_embed,inp_layer


def wide_deep(df_train,df_test,wide_cols,x_cols,embedding_cols,cont_cols,method,target):

    """
        Run the wide and deep model. Parameters are the same as those for the wide and deep function
        wide and deep function
    """

    X_train_wide,y_train_wide,X_test_wide,y_test_wide = wide(df_train,df_test,wide_cols,x_cols,target,model_type,method)
    X_train_deep,y_train_deep,X_test_deep,y_test_deep,deep_inp_embed,deep_inp_layer = deep(df_train,df_test,embedding_cols,cont_cols,target,model_type,method)


    X_tr_wd = [X_train_wide] + X_train_deep
    Y_tr_wd = y_train_deep #wide or deep is the same here
    X_te_wd = [X_test_wide] + X_test_deep
    Y_te_wd = y_test_deep

    activation,loss,metrics = fit_param[method]
    if metrics:metrics = [metrics]

    #WIDE
    w = Input(shape=(X_train_wide.shape[1],),dtype='float32',name='wide')

    #DEEP: the output of the 50 neurons layer will be the deep-side input
    d = concatenate(deep_inp_embed)
    d = Flatten()(d)
    d = Dense(50,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.02))(d)
    d = Dropout(0.5)(d) # Dropout don't seem to help in this model
    d = Dense(20,activation='relu',name='deep')(d)
    d = Dropout(0.5)(d)

    #WIDE + DEEP
    wd_inp = concatenate([w,d])
    wd_out = Dense(Y_tr_wd.shape[1],activation=activation,name='wide_deep')(wd_inp)
    wide_deep = Model(inputs=[w] + deep_inp_layer,outputs=wd_out)
    wide_deep.compile(optimizer='Adam',loss=loss,metrics=metrics)
    wide_deep.fit(X_tr_wd,Y_tr_wd,epochs=10,batch_size=128)

    #Maybe you want to schedule a second search with lower learning rate
    wide_deep.optimizer.lr = 0.0001
    wide_deep.fit(X_tr_wd,Y_tr_wd,epochs=10,batch_size=128)

    results = wide_deep.evaluate(X_te_wd,Y_te_wd)
    print("\n",results)
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
        wide_deep(df_train,df_test,wide_cols,x_cols,embedding_cols,cont_cols,method,target)
    





    print("method:",method)
    print("model_type:",model_type)
    print("train_data:",train_data)
    print("test_data",test_data)


