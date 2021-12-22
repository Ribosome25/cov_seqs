"""

"""
import os
import matplotlib.pyplot as plt
import numpy as np
from ref_models.args import VHSE_CNN_training_Args
from ref_models.cnn_model import build_VHSE_CNN
from topo_reg.io import check_path_exists, load_int_dict_from_json, parent_dir, read_df_list, save_df
from sklearn.model_selection import train_test_split
import tensorflow as tf


def plot_loss(history, savedir):
    try:
        plt.plot(history.history['loss'], label='loss')
    except KeyError:
        pass
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'train_loss.png'))


def train_VHSE_CNN(args: VHSE_CNN_training_Args):
    xx_df = read_df_list(args.x_train)
    yy = read_df_list(args.target_file)[args.targets]

    xx = xx_df.values.copy()
    xx.resize((len(xx_df), args.max_length * 8))  # pad with 0s
    xx = xx.reshape(len(xx_df), -1, 8)

    X_train, X_valid, y_train, y_valid = train_test_split(xx, yy, test_size=0.1, random_state=args.seed)
    params = load_int_dict_from_json(args.model_params)
    model = build_VHSE_CNN(max_length=args.max_length, **params)
    # model = build_VHSE_CNN(max_length=args.max_length, n_chnls=(256, 0), kers_strides=(8, 4, 8, 4), opt='adam', layer_config='1111')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    
    print(X_train.shape)
    print(model.layers[0].input_shape)
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=200,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_valid, y_valid.values),
        callbacks=[callback]
    )

    check_path_exists(args.save_dir)
    # save_dir = parent_dir(args.save_dir)
    save_dir = args.save_dir
    np.save(os.path.join(save_dir, "valid_prediction.npy"), model.predict(X_valid))
    valid_err = history.history['val_loss'][-1]
    print("Valid Error.", valid_err)

    def myprint(s):
        with open(os.path.join(save_dir, "model_summary.txt"),'a') as f:
            print(s, file=f)

    model.summary(print_fn=myprint)
    if args.plot_history:
        plot_loss(history, save_dir)
    model.save(os.path.join(save_dir, "model.h5"))

    if args.cross_valid:
        print("cv TBD")
