import tensorflow
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from numpy.random import seed
from tensorflow import set_random_seed
from trainer.checkpointers import *
from trainer.utils import *
from trainer.accuracy import *
from tensorflow.python.keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

seed(11)
set_random_seed(12)

job_dir = 'output'
data_path = 'dataset'
train_csv = "tops_train_shuffle.csv"
val_csv="tops_val_full.csv"
train_epoch = 25
batch_size = 16
lr = 0.001
img_width, img_height = 224, 224
batch_size *= 3

def accuracy_function(y_true, y_pred):
    def euclidian_dist(p, q):
        return K.sqrt(K.sum((p-q)**2))
    
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    accuracy = 0
    
    for i in range(0, batch_size, 3):
        try:
            query_embed = y_pred[i+0]
            pos_embed = y_pred[i+1]
            neg_embed = y_pred[i+2]
            dist_query_pos = euclidian_dist(query_embed, pos_embed) 
            dist_query_neg = euclidian_dist(query_embed, neg_embed)
            accuracy += tf.cond(dist_query_neg > dist_query_pos, lambda:1, lambda:0)
        except:
            continue
    accuracy = tf.cast(accuracy, tf.float32)
    
    return accuracy*100/(batch_size/3)

def loss_function(y_true, y_pred):
    def euclidian_dist(p, q):
        return K.sqrt(K.sum((p - q)**2))
    def contrastive_loss(y, dist):
        one = tf.constant(1.0, shape=[1], dtype=tf.float32)
        square_dist = K.square(dist)
        square_margin = K.square(K.maximum(one - dist, 0))
        return K.mean(y * square_dist + (one - y) * square_margin)

    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    pos = tf.constant(1.0, shape=[1], dtype=tf.float32)
    neg = tf.constant(0.0, shape=[1], dtype=tf.float32)

    for i in range(0, batch_size, 3):
        try:
            query_embed = y_pred[i+0]
            pos_embed = y_pred[i+1]
            neg_embed = y_pred[i+2]
            dist_query_pos = euclidian_dist(query_embed, pos_embed)
            dist_query_neg = euclidian_dist(query_embed, neg_embed)
            loss_query_pos = contrastive_loss(pos, dist_query_pos)
            loss_query_neg = contrastive_loss(neg, dist_query_neg)
            loss = (loss + loss_query_pos + loss_query_neg)
        except:
            continue
    loss *= 1/(batch_size*2/3)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss, zero)

def ranknet():
    #ranknet: uses pre-trained VGG19 + 2 shallow CNNs
    vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Lambda(lambda x: K.l2_normalize(x, axis=1))(convnet_output)

    s1 = MaxPool2D(pool_size=(4,4), strides=(4,4), padding='valid')(vgg_model.input)
    s1 = ZeroPadding2D(padding=(4,4), data_format=None)(s1)
    s1 = Conv2D(96, kernel_size=(8,8), strides=(4,4), padding='valid')(s1)
    s1 = ZeroPadding2D(padding=(2,2), data_format=None)(s1)
    s1 = MaxPool2D(pool_size=(7,7), strides=(4,4), padding='valid')(s1)
    s1 = Flatten()(s1)

    s2 = MaxPool2D(pool_size=(8,8), strides=(8,8), padding='valid')(vgg_model.input)
    s2 = ZeroPadding2D(padding=(4,4), data_format=None)(s2)
    s2 = Conv2D(96, kernel_size=(8,8), strides=(4,4), padding='valid')(s2)
    s2 = ZeroPadding2D(padding=(1,1), data_format=None)(s2)
    s2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(s2)
    s2 = Flatten()(s2)

    merge_one = concatenate([s1, s2])
    merge_one_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(merge_one)
    merge_two = concatenate([merge_one_norm, convnet_output], axis=1)
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(inputs=vgg_model.input, outputs=l2_norm_final)

    return final_model

model = ranknet()
dg = DataGenerator({
    "rescale": 1. / 255,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.2,
    "rotation_range": 30
}, data_path, train_csv, val_csv, target_size=(img_width, img_height))

train_generator = dg.get_train_generator(batch_size, False)
test_generator = dg.get_test_generator(batch_size)

_loss_tensor = loss_function
accuracy = accuracy_function
model.compile(loss=_loss_tensor, optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True), metrics=[accuracy])
model_checkpoint_path = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
model_checkpointer = ModelCheckpoint(job_dir, model_checkpoint_path, save_best_only=True, save_weights_only=True,
                                     monitor="val_loss", verbose=1)
tensorboard = TensorBoard(log_dir=job_dir + '/logs/', histogram_freq=0, write_graph=True, write_images=True)
callbacks = [model_checkpointer, tensorboard]

model.fit_generator(train_generator,
                    steps_per_epoch=(train_generator.n // (train_generator.batch_size)),
                    validation_data=test_generator,
                    epochs=train_epoch,
                    validation_steps=(test_generator.n // (test_generator.batch_size)),
                    callbacks=callbacks)

model.save_weights('output/model.h5')
