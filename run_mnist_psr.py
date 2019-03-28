import os
from os.path import isdir, join
import timeit
import argparse

#from sklearn.metrics import accuracy_score

# avoid the odd behavior of pickle by importing under a different name
import pcanet_based as net
from utils import load_model, save_model, load_mnist, set_device

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

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
    print("Finding PCA filters took %.1f seconds"%(train_time,))


    #Now we have the filters, apply them to a couple of random test images
    #and calculate and display the resulting PSRs
    for i in np.random.choice(10000, 5):
        imagePSR = pcanet.create_PSR(test_set[0][i])

        plt.imshow(imagePSR[0], cmap='gray')
        plt.show()
    
    return pcanet


def test(pcanet, pcanet_m2, test_set):
    """
    A simple test. Take a random image from the test set, rotate by a random 
    amount, get the PSRs for both and show them. 
    
    To add later: Calculate and minimize the difference between the PSRs by
    L-BFGS optimization
    """
    idx = np.random.choice(10000)
    image_orig = test_set[0][idx]

    plt.imshow(np.asarray(image_orig[...,0]), cmap='gray', vmin=0, vmax=1)
    plt.title("Original image of digit %d"%(test_set[1][idx],))
    plt.show()

    imagePSR = pcanet.create_PSR(image_orig)
    plt.imshow(imagePSR[0], cmap='gray', vmin=0, vmax=1)
    plt.title("PSR of original image")
    plt.show()


    adjimage = modality_change(image_orig[...,0])
    rot = np.random.random()/5. - 0.1 #random rotation between -10 and +10 percent
   
    adjimage_rot = adjimage*255.0
    adjimage_rot = Image.fromarray(adjimage_rot.astype('uint8'), mode='L')
    adjimage_rot = adjimage_rot.rotate(rot*360)
    adjimage_rot = np.asarray(adjimage_rot)/255.0
    
    plt.imshow(adjimage_rot, cmap='gray', vmin=0, vmax=1)
    plt.title("Intensity changed and rotated by %2.1f percent"%(rot*100,))
    plt.show()

    rotPSR = pcanet_m2.create_PSR(adjimage_rot)
    plt.imshow(rotPSR[0], cmap='gray', vmin=0, vmax=1)
    plt.title("PSR of rotated image")
    plt.show()
    
    outputImagePSR = (imagePSR[0]*255.0).astype('uint8')
    outputRotPSR = (rotPSR[0]*255.0).astype('uint8')
    imageio.imwrite('psr_orig_%d.png' % (idx,), outputImagePSR)
    imageio.imwrite('psr_rot%2.1f_%d.png' % (rot*100,idx,), outputRotPSR)

    return

def modality_change(image):
    """
    Synthetically change the modality of an image in a non-linear way, just as
    an initial test of the method
    """
    maxInt = np.max(image)
    minInt = np.min(image)
    
    #rebase to zero
    newimage = image - minInt
    #Shift intensity by 10%
    newimage = newimage - (maxInt-minInt)/10
    #Square it for non-linearity (and low 10% of values flipped)    
    newimage = pow(newimage, 2)
    #Rescale so values lie no higher than original max level
    newimage = newimage / (maxInt - minInt)
    
    return newimage
    

train_set, test_set = load_mnist()
train_set = (train_set[0][:1000],train_set[1][:1000])

if args.gpu >= 0:
    set_device(args.gpu)


if args.mode == "train":
    print("Training the model...")
    pcanet = train(train_set)

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet.pkl"))
    #save_model(PSR, join(args.out, "PSR.pkl"))
#    print("Model saved")
    print("Training the model with synthetic modality...")
    pcanet = train([modality_change(train_set[0]),train_set[1]])

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet_m2.pkl"))

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
    pcanet_m2 = load_model(join(args.pretrained_model, "pcanet_m2.pkl"))
#    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

#    y_test, y_pred = test(pcanet, classifier, test_set)
    test(pcanet, pcanet_m2, test_set)

#    accuracy = accuracy_score(y_test, y_pred)
#    print("accuracy: {}".format(accuracy))
#    print("Test mode not implemented")
