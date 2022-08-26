import os
import pandas as pd
import tensorflow.compat.v1 as tf 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
#导入数据包


def build_model_columns():
    #特征处理：连续特征、离散特征、转换特征、交叉特征

    #连续特征
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('captial_gain')
    capital_loss = tf.feature_column.numeric_column('captical_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    #离散特征
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',[
            '10th','11th','12th','1st-4th','5th-6th','7th-8th','9th','Assoc-acdm',
            'Assoc-voc','Bachelors','Doctorate','HS-grad','Masters','Preschool','Prof-school',
            'Some-college'
        ]
    )
    
    maritial_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',[
            'Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married',
            'Separated','Widowed'
        ]
    )

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',[
            'Husband','Not-in-family','Other-relative','Own-child','Unmarried','Wife'
        ]
    )

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',[
            '?','Federal-gov','Local-gov','Never-worked','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay'
        ]

    )

    #离散hash bucket特征
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation',hash_bucket_size = 1000
    )

    #特征trasformations
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries = [18,25,30,35,40,45,50,55,60,65]
    )


    #2. 设定wide层特征
    """
    wide 部分使用了规范化后的连续特征、离散特征、交叉特征
    """

    #基本特征列
    base_columns = [
        #全局离散特征
        education,maritial_status,relationship,
        workclass,occupation,age_buckets,
    ]


    #交叉特征列
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education','occupation'],hash_bucket_size=10000
        ),
        tf.feature_column.crossed_column(
            [age_buckets,'education','occupation'],hash_bucket_size =1000
        )
    ]

    #wide特征列
    wide_columns = base_columns + crossed_columns



    #设定deep层特征
    deep_columns=[
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),

        #embedding特征
        tf.feature_column.embedding_column(occupation,dimension=8)
    ]

    return wide_columns,deep_columns



#数据准备
def get_data_download(train_data,test_data):
    """if adult data "train.csv" and "test.csv" are not in your directory,
    download them.
    """
   

    if not os.path.exists(train_data):
        print("downloading training data....")
        df_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",names=COLUMNS, skipinitialspace=True)
        df_train.to_csv(train_data)
    else:
        df_train = pd.read_csv(train_data)


    if not os.path.exists(test_data):
        print("downloading testing data...")
        df_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",names=COLUMNS,skipinitialspace=True)
        df_test.to_csv(test_data)
    else:
        df_test = pd.read_csv(test_data)
    print(df_train.filter(items=['age','education_num','capital_gain','capital_loss','occupation']).head(10))
    print(df_test.filter(items=['age','education_num','capital_gain','capital_loss','occupation']).head(10))
    print(df_train.groupby(['income_bracket']).count())
    return df_train,df_test


#定义输入
def input_fn(data_file,num_epochs,shuffle,batch_size):
    """为estimator创建一个input function"""
    #assert判断，为Fasle时执行后面的语句
    assert tf.io.gfile.exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing",data_file)
        #tf.decode_csv 会把csv文件转换成Tesnsor，其中record_defaults用于指明每一列缺失值的填充。
        columns = tf.io.decode_csv(line,record_defaults=COLUMNS_DEFAULTS)
        features = dict(zip(COLUMNS,columns))
        labels = features.pop('income_bracket')
        return features,tf.equal(labels,'>50K')

    dataset = tf.data.TextLineDataset(data_file).map(parse_csv,num_parallel_calls=1)

    """
    tf.data.DataSet.map 我们可以很方便地对数据集中各元素进行预处理。
    map接收一个函数，DataSet中的每个元素都会被当做这个函数的输入
    num_parallel_calls 取决于你的硬件，训练数据的特质，比如，size，shape
    CPU 有4个核时，将num_parallel_calls 设置为4将会更高效

    也可以设置shuffle，shuffle功能打乱dataset的元素，它有一个参数buffersize，表示打乱使用的buffer的大小，建议舍的不要太小，一般设置为1000
    """

    #repeat功能是将整个序列重复多次，主要是用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat（2）就可以将之变成2个epoch
    #dataset = dataset.repeat(num_epochs)
    #设置数据集的批量处理大小
    #dataset = dataset.batch(batch_size)

    #iterator = dataset.make_one_shot_iterator()
    #batch_features,batch_labels = iterator.get_next()
    #return batch_features,batch_labels



#模型准备
# wide & deep Model
def build_estimator(model_dir,model_type):
    #定义特征输入，处理和离散
    wide_columns,deep_columns = build_model_columns()
    hidden_units = [100,50]
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU
    # trains faster than GPU for thie model
    """
    run_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU':0})
    )
    """

    if model_type == "wide":
        return tf.estimator.LinearClassifier(
            model_dir = model_dir,
            feature_columns = wide_columns,
            config = run_config
        )
    if model_type == "deep":
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_uints,
            config=run_config
        )
    else:
        return tf.estimator.DNNlinerCombinedClassifier(
            model_dir=model_dir,
            line_feature_columns = wide_columns,
            dnn_feature_columns = deep_columns,
            dnn_hidden_units = hidden_uints,
            config = run_config
        )





#模型结果输出



if __name__ == "__main__":
    stage = 3 #1. 数据下载 3. test the function of input_fn 2. train model 
    train_data = "/root/along/deepLearning/widedeep/data/train.csv"
    test_data = "/root/along/deepLearning/widedeep/data/test.csv"
    if stage == 1:
        get_data_download(train_data,test_data)

    #test the function of input_fn
    if stage == 3: 
        input_fn(train_data,10,True,100)


    #模型训练
    if stage == 2:
        model_type = 'widedeep'
        model_dir = 'xxx'

        #wide & deep 联合模型
        model = build_estimator(model_dir,model_type)
        train_epochs = 10
        batch_size = 5000
        train_file = "xxx"
        test_file = "xxx"
        for n in range(train_epochs):
            #模型训练
            model.train(input_fn=lambda:input_fn(train_file,train_epochs,True,batch_size))
            #模型预估
            results = model.evaluate(input_fn = lambda:input_fn(test_file,1,False,batch_size))
            #打印评估结果
            print("Result at epoch {0}".format((n+1)*train_epochs))
            print('-'*30)
            for key in sorted(results):
                print("{0:20}:{1:.4f}".format(key,results[key]))



