# save_augmentation_image_by_folder
This is the code for svae the Image generate by the ImageDataGenerator, the file will save follow by your origin file tree.

Before using please install the following package:
scipy
tensorflow
keras
numpy
scipy
Pillow

Using the test_set for example:

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/File%20path.png)

It has two class: cats, dogs

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/test%20set.png)

After running the program:

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/Running%20record.png)

We can have the new data

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/Output%20data.png)

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/example%20image.png)

The output number of the image can be calculate by:

Number in origin file* rate* batch_size* range

![image](https://github.com/reece762/save_augmentation_image_by_folder/blob/master/setting.png)
