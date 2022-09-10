# save_augmentation_image_by_folder
This is the code for svae the Image generate by the ImageDataGenerator, the file will save follow by your origin file tree.

Why for doing this? 

It is so slow when using ImageDataGenerator: 3 second for a epochs > 60 second for a epochs...  <br />
Also some image after augmentation it is being a no use data, so saving the image can let me traning faster and the quality of image will be better after human filtering.

Before using please install the following package:<br />
scipy<br />
tensorflow<br />
keras<br />
numpy<br />
scipy<br />
Pillow<br />

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
