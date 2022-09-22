import numpy as np
import pandas as pd
from essm import CTCVRNet
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import time 

#from model_train import train_model

#build the generate of data(train)
ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000,5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0,10,size=(10000,5)),columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0,10,size=(10000,3)),columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000,5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0,10,size=(10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0,10,size=(10000,3)),columns=['item_cate_{}'.format(i) for i in range(3)])

#build the generate of data(val)
ctr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000,5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_val = pd.DataFrame(np.random.randint(0,10,size=(10000,5)),columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_val = pd.DataFrame(np.random.randint(0,10,size=(10000,3)),columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000,5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_val = pd.DataFrame(np.random.randint(0,10,size=(10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000,5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_val = pd.DataFrame(np.random.randint(0,10,size=(10000,3)),columns=['item_cate_{}'.format(i) for i in range(3)])


#build the target of train
ctr_target_train = pd.DataFrame(np.random.randint(0,2,size=10000))
cvr_target_train = pd.DataFrame(np.random.randint(0,2,size=10000))

#build the target of val
ctr_target_val = pd.DataFrame(np.random.randint(0,2,size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0,2,size=10000))

cate_feature_dict = {}
user_cate_feature_dict ={}
item_cate_feature_dict = {}

for idx,col in enumerate(ctr_user_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
    user_cate_feature_dict[col] = (idx,ctr_user_cate_feature_train[col].max()+1)

for idx,col in enumerate(ctr_item_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_item_cate_feature_train[col].max()+1
    item_cate_feature_dict[col] = (idx,ctr_item_cate_feature_train[col].max() +1)



print(cate_feature_dict)
print(user_cate_feature_dict)
print(item_cate_feature_dict)

ctcvr = CTCVRNet(cate_feature_dict)
ctcvr_model = ctcvr.build(user_cate_feature_dict,item_cate_feature_dict)
opt = optimizers.Adam(lr=0.003,decay=0.0001)
ctcvr_model.compile(optimizer=opt,loss=['binary_crossentropy','binary_crossentropy'],loss_weight=[1.0,1.0],metrics=[tf.keras.metrics.AUC()])

#keras model save path
filepath = 'esmm_best.h5'

#call back function
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=2,min_lr=0.0001,verbose=1)
earlystopping = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=8,verbose=1,mode='auto')
callbacks = [checkpoint,reduce_lr,earlystopping]


#model_train
ctcvr_model.fit([ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
	                 ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
	                 cvr_item_numerical_feature_train,
	                 cvr_item_cate_feature_train], [ctr_target_train, cvr_target_train], batch_size=256, epochs=50,
	                validation_data=(
		                [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
		                 ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
		                 cvr_item_numerical_feature_val,
		                 cvr_item_cate_feature_val], [ctr_target_val, cvr_target_val]), callbacks=callbacks,
	                verbose=0,
	                shuffle=True)
#load model and save as tf_serving model
saved_model_path = './esmm/{}'.format(int(time.time()))
ctcvr_model = tf.keras.models.load_model('esmm_best.h5')
tf.saved_model.save(ctcvr_model,saved_model_path)