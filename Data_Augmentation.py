import PIL
from keras.preprocessing.image import ImageDataGenerator
import os, random
import numpy as np



def train_data(fileDir, target_dir):

    pathDir = os.listdir(fileDir)  # read data path

    #More information for ImageDataGenerator can see in here https: // www.tensorflow.org / api_docs / python / tf / keras / preprocessing / image / ImageDataGenerator
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    for i in range(len(pathDir)):
        newdir = os.path.join(fileDir, pathDir[i])
        newpathDir = os.listdir(newdir)
        targetdir = os.path.join(target_dir, pathDir[i])
        filenumber = len(newpathDir)
        print("newdir:", newdir)
        print("targetdir:", targetdir)
        print("filenumber:", filenumber)
        if not os.path.isdir(targetdir):
            os.makedirs(targetdir, exist_ok=True)

        rate = 0.5  # Process ratio, it mena how many % of image in the file you want to process
        picknumber = int(filenumber * rate)
        sample = random.sample(newpathDir, picknumber)  # random select image
        for name in sample:
            image = np.expand_dims(PIL.Image.open(newdir + '/' + name), 0)

            image_gen_train.fit(image)
            for x, val in zip(image_gen_train.flow(image,  # image we chose
                                                   save_to_dir=targetdir,  # this is where we figure out where to save
                                                   save_format='png'), range(
                3)):  # here we define a range because we want 10 augmented images otherwise it will keep looping forever
                # I think
                continue

        # for i in range(2):
        #     train_data_gen.next()
    print('\n')
    print('Train data augmentation complete')


def val_data(val_dir, target_size1, target_size2):
    image_gen_val = ImageDataGenerator(rescale=1. / 255)

    val_data_gen = image_gen_val.flow_from_directory(
        directory=val_dir,
        target_size=(target_size1, target_size2),
        class_mode='sparse'
    )

    print('Val data load complete')
    return val_data_gen


if __name__ == '__main__':
    fileDir = "./train/"  # source data path
    save_dir = "./train/"
    train_data(fileDir, save_dir)
    quit()
