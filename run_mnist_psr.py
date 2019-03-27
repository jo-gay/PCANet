import os
from os.path import isdir, join
import timeit
import argparse

from sklearn.metrics import accuracy_score

# avoid the odd behavior of pickle by importing under a different name
import pcanet_based as net
from utils import load_model, save_model, load_mnist, set_device

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="PSR example")
parser.add_argument("--gpu", "-g", type=int, default=-1,
                    help="GPU ID (negative value indicates CPU) NB UNTESTED")

subparsers = parser.add_subparsers(dest="mode",
                                   help='Choice of train/test mode')
subparsers.required = True
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--out", "-o", default="result",
                          help="Directory to output the result")

test_parser = subparsers.add_parser("test")
test_parser.add_argument("--pretrained-model", default="result",
                         dest="pretrained_model",
                         help="Directory containing the trained model")

args = parser.parse_args()


def train(train_set):
    images_train, y_train = train_set

    print("Training PCANet")

    pcanet = net.PCANetBased(
        image_shape=28,
        filter_shape_l1=3, step_shape_l1=1, n_l1_output=8,
        filter_shape_l2=3, step_shape_l2=1, n_l2_output=8,
        filter_shape_pooling=2, step_shape_pooling=2
    )

    pcanet.validate_structure()

#Fit calculates the PCA filters for both layers, stored in the PCANet object
    t1 = timeit.default_timer()
    pcanet.fit(images_train)
    t2 = timeit.default_timer()

    train_time = t2 - t1
    print("Finding PCA filters took %r"%(train_time,))


    #Now we have the filters, display the resulting PSRs for a couple of
    #random test images
    for i in np.random.choice(10000, 5):
        imagePSR = pcanet.create_PSR(test_set[0][i])

        plt.imshow(imagePSR[0])
        plt.show()
    
    return pcanet


def test(pcanet, classifier, test_set):
    images_test, y_test = test_set

    X_test = pcanet.transform(images_test)
    y_pred = classifier.predict(X_test)
    return y_pred, y_test


train_set, test_set = load_mnist()
train_set = (train_set[0][:10000],train_set[1][:10000])

if args.gpu >= 0:
    set_device(args.gpu)


if args.mode == "train":
    print("Training the model...")
    pcanet = train(train_set)

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet.pkl"))
    #save_model(PSR, join(args.out, "PSR.pkl"))
    print("Model saved")

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
#    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

#    y_test, y_pred = test(pcanet, classifier, test_set)

#    accuracy = accuracy_score(y_test, y_pred)
#    print("accuracy: {}".format(accuracy))
    print("Test mode not implemented")
