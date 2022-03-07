
import numpy as np
from numpy.lib.function_base import append
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from logManager import logFile_update
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight 
import datasetManager as aloe
import csv
import cv2 as cv

K.set_image_data_format('channels_last')

IMG_SIZE=32

def CapsNet(input_shape, n_class, routings, batch_size):
    """
    Capsule network for Aloe Vera Dataset.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, one for training, one for evaluation.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: Conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=9, strides=1, padding='valid', activation=layers.LeakyReLU(alpha=0.3), name='conv1')(x)
    #filters=256

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=4, n_channels=8, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Dynamic Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings, name='aloecaps')(primarycaps)


    # Layer 4: This layer replaces each capsule with its length. Just to match the true label's shape.
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    #Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(64, activation=layers.LeakyReLU(alpha=0.3), input_dim=8 * n_class))
    decoder.add(layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3)))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 8)) #16
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss, same equation from Hinton's paper.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model,  # type: models.Model
          data, args):
    """
    For Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv', append=True)
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    lr_decay = callbacks.ReduceLROnPlateau(
    monitor="val_capsnet_accuracy",
    factor=args.lr_decay,
    patience=3,
    verbose=1,
    mode="max",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    # # Training without data augmentation:
    # model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
    #           validation_data=([x_test, y_test], [y_test, x_test]), callbacks=[log, tb, checkpoint, lr_decay])
 

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)

    # Training with data augmentation. If shift_fraction=0., no augmentation.
    model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay, tb])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    #model.save(args.save_dir + '/my_capsnet')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    logFile_update("./result/log.csv")
    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def test(model, data, args, full_res_data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size) #100
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    
    #x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.argmax(y_test, axis=1)
    y_classes = [np.argmax(element) for element in y_pred]
    print("Classification Report: \n", classification_report(y_test, y_classes))
    conf_matrix = confusion_matrix(y_true = y_test, y_pred = y_classes)
    print("Confusion Matrix:\n",conf_matrix)

    classification = ["Healthy", "Rot", "Rust"]

    ### Writes all the classification of each image from the test data in a csv file.###
    # for element in range(len(y_classes)):#len(y_classes)
    #     with open('allClassifications.csv', 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([classification[y_classes[element]]])

def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            #x_recon = model.predict([x, y, tmp])
            x_recon = model.predict([x, y, tmp], batch_size=1)
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[:100]
    y_train = y_train[:100]
    x_test = x_test[:100]
    y_test = y_test[:100]

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_aloe():
    (x_train, y_train) = aloe.load_training_data()
    (x_test, y_test) = aloe.load_test_data()
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Aloe Vera Dataset.")
    parser.add_argument('--epochs', default=200, type=int) 
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0") 
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=1, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_aloe()
    (x_test_full, y_test_train) = aloe.load_test_data_fullsize()
    
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)
    model.summary()

    # train or test
    if args.weights is not None:  # initialize the model weights with provided one
        model.load_weights(args.weights)
    if os.path.exists("./result/trained_model.h5"):
        print("trained model found!")
        print("loading weights from ./result/trained_model.h5")
        model.load_weights("./result/trained_model.h5")
        print("weights succesfully loaded!")
    if not args.testing:
        #train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        test(model=eval_model, data=(x_test, y_test), args=args, full_res_data=(x_test_full, y_test_train))
    else:  # as long as weights are given, testing will run
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)