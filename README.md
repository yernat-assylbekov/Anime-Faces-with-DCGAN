# Anime-Faces-with-DCGAN

In this repository we generate Anime Face images with Generative Adversarial Network (GAN). This is my first experience with GAN.

My implementation uses Python 3.6.7, TensorFlow 2.0, Numpy and Matplotlib.

## Dataset

The dataset was downloaded from https://www.kaggle.com/soumikrakshit/anime-faces which consists of 21551 anime faces images of sizes 64x64. Here are sample images from the dataset

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/images_from_train_set.png?raw=true)

## Network Architecture

I use a slightly modified DCGAN (Deep Convolutional GAN) and follow the guidlines listed below:<br>
• Replace all max pooling with convolutional strides.<br>
• Use transposed convolution for upsampling.<br>
• Use batchnorm in both the generator and the discriminator.<br>
• Use ReLU activation in the generator for all layers except for the output layer.<br>
• Use LeakyReLU activation in the discriminator for all layers except for the flattening layer and the output layer.<br>
• The output layers of both the generator and the discriminator use sigmoid activation.

The precise architectures for the generator and the discriminator are as shown below.<br>
### The generator:<br>
![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/generator_diagram.png?raw=true)<br>
### The discriminator:<br>
![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/discriminator_diagram.png?raw=true)

## Training Details

I used the same loss functions for the generator and the discriminator as in the last paragraph of Section 3 in the GAN paper <a href="https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf">[1]</a> by Goodfellow et al. We use the Adam optimizer with `learning_rate = 0.0001` and `beta_1 = 0.5`. I trained the model with an NVIDIA K80 GPU.

## Results

After training for 500 epochs, I was satisfied with the results. Below is the full training as a GIF with images sampled every 10 epochs.

![alt text](https://github.com/yernat-assylbekov/Anime-Faces-with-DCGAN/blob/master/images/anime_faces_generated.gif?raw=true)

I expect that the results will be improved if one trains longer.

## References

<a href="https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf">[1]</a> I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville and Y. Bengio, <i>Generative Adversarial Nets</i>, NIPS Proceedings (2014).
