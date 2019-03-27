import os
from os.path import isdir, join
import timeit
import argparse

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# avoid the odd behavior of pickle by importing under a different name
import pcanet_based as net
from utils import load_model, save_model, load_mnist, set_device

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PCANet example")
parser.add_argument("--gpu", "-g", type=int, default=-1,
                    help="GPU ID (negative value indicates CPU)")

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

#JCG comment: Fit calculates the PCA filters for both layers, stored in the PCANet object
    t1 = timeit.default_timer()
    pcanet.fit(images_train)
    t2 = timeit.default_timer()

    train_time = t2 - t1
    print("Finding PCA filters took %r"%(train_time,))


#Now we have the filters, apply them to the test/target images and use the 
#output to create the PSR

    t1 = timeit.default_timer()
    image0PSR = pcanet.create_PSR(test_set[0][0])
    image1PSR = pcanet.create_PSR(test_set[0][1])
    t2 = timeit.default_timer()

    transform_time = t2 - t1
    print("Creating PSRs took %r"%(transform_time,))

#JCG comment: Transform applies the filters to the training data and returns the 
#output feature maps from layers 1 and 2
#    l1out, l2out = pcanet.transform(images_train[0])

#JCG comment: fuse_maps turns L1 feature maps into a single fused image
#    t1 = timeit.default_timer()
#    F1 = pcanet.fuse_maps(l1out)
#    F2 = pcanet.fuse_maps(l2out)
#    t2 = timeit.default_timer()
#
#    fuse_time = t2 - t1
#    print("Creating fused feature maps took %r"%(fuse_time,))

#JCG comment: create_PSR turns L1 and L2 fused feature maps into the final PSR
#    t1 = timeit.default_timer()
#    PSR = pcanet.create_PSR(F1, F2)
#    t2 = timeit.default_timer()
#
#    psr_time = t2 - t1
#    print("Creating fused feature maps took %r"%(psr_time,))

    plt.imshow(image0PSR[0])
    plt.show()
    plt.imshow(image1PSR[0])
    plt.show()

#    print("Training the classifier")
#
#    t1 = timeit.default_timer()
#    classifier = SVC(C=10)
#    classifier.fit(X_train, y_train)
#    t2 = timeit.default_timer()
#    print("Training the SVC took %r"%(t2-t1,))
    
    return pcanet, PSR


def test(pcanet, classifier, test_set):
    images_test, y_test = test_set

    X_test = pcanet.transform(images_test)
    y_pred = classifier.predict(X_test)
    return y_pred, y_test


train_set, test_set = load_mnist()
train_set = (train_set[0][:1000],train_set[1][:1000])
test_set = (test_set[0][:200],test_set[1][:200])

if args.gpu >= 0:
    set_device(args.gpu)


if args.mode == "train":
    print("Training the model...")
    pcanet, PSR = train(train_set)

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet.pkl"))
    #save_model(PSR, join(args.out, "PSR.pkl"))
    print("Model saved")

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

    y_test, y_pred = test(pcanet, classifier, test_set)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: {}".format(accuracy))
