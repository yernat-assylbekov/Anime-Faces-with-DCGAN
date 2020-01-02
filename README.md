# Anime-Faces-with-DCGAN

In this repository we generate Anime Face images with Generative Adversarial Network (GAN). This is my first experience with GAN.

My implementation uses Python 3.6.7, TensorFlow 2.0, Numpy and Matplotlib.

The dataset was downloaded from https://www.kaggle.com/soumikrakshit/anime-faces which consists of 21551 anime faces images of sizes 64x64. Here are sample images from the dataset

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/images_from_train_set.png?raw=true)

I use a slightly modified DCGAN (Deep Convolutional GAN). The architectures for the generator and the discriminator are as shown below.

The generator:

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/generator_diagram.png?raw=true)

The discriminator:

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/discriminator_diagram.png?raw=true)

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/anime_faces_generated.gif?raw=true)
