# Session-2 | EVA4P2
#### SANE_NEURONS (https://www.saneneurons.com)

### Assignment:
1. Collect 1000 images for one of the classes mentioned below:
- FlyingBirds
- LargeQuadCopters
- SmallQuadCopters
- WingedDrones (assigned to our group)

    *In all of these images, the objects MUST be flying and not on the ground (so no product images or them on the ground). You can use Google, Flickr, Bing, Yahoo, or DuckDuckGo for image search.*

2. Train (transfer learning) MobileNet_V2 on a custom dataset of 31000(or total images collected for all the classes) images
- Make 70:30 split for train/validation image dataset.
- Name the 4 classes as **FlyingBirds, LargeQuadCopters, SmallQuadCopters** and **WingedDrones**.
- MobileNet_V2 is trained on 224 x 224 image size, the downloaded images from web might not be of same size. Think and implement best strategy.
- Save the model and upload it on AWS Lambda. Keep it ready for future use (use the same S3 bucket as Session-1)
- Submit the answers in S2 Quiz.

### Solution:
#### Code Explanation
In this section, we briefy walk you through the Python Notebook for the training:
- First, import the necessary modules from **torch, torchvision** and **matplotlib** libraries.
- In the next step we load the image dataset from the shared Drive and split them with 70:30 ratio. We also show their stats like count of each class.
- We then transfer the image files according to the MobileNet folder requirements (e.g. /train/class_name/ and /val/class_name/).
- After transforming the Images using **torchvision.transforms** (e.g. *Rotation, Crop, Image Flip, Random Erasing*) we traing the model with transformed images and calculate the Running Loss.
- After training (and subsequent validation) we plot the accuracy graph for Training and Validation step.
- We also show the misclassified samples from each of the 4 classes.
- Finally we export the model file (to be used in future sessions with AWS Lambda).

#### Datasets
This [Google Drive folder](https://drive.google.com/drive/folders/1co2Ik7knQLrrDf7hqo0-TaYaolr1dpDw) has images from all 4 classes.

#### Resizing Strategy
We have used the open-source GIMP (GNU Image Manipulation Program) image editor tool to resize the images and to convert all images to JPEG format. Used BIMP (Batch Image Manipulation Plugin) for GIMP to perform batch manipulation of Images.

#### Model Brief
We used the pre-trained PyTorch ***MobileNet_V2*** model and **Trained** the model(transfer learning) according to our 4 image classes. (4 features classifier). Here's the brief summary of the model architecture.
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/mobilenet_v2_architecture.png "MobileNet V2 Architecture")

As you can see above *MobileNet V2* extends its predecessor ([MobileNet](https://arxiv.org/abs/1704.04861)) with two new ideas.
##### Inverted Residuals
As oppose to how original **[Residual Connection](https://arxiv.org/abs/1512.03385)** works, Inverted Residuals of MobileNet V2 follows the narrow -> wide -> narrow approach w.r.t. number of channels. The first step widens the network using a 1x1 convolution because the following 3x3 depthwise convolution already greatly reduces the number of parameters. Afterwards another 1x1 convolution squeezes the network in order to match the initial number of channels.

##### Linear Bottlenecks
The reason we use non-linear activation functions in neural networks is that multiple matrix multiplications cannot be reduced to a single numerical operation. It allows us to build neural networks that have multiple layers. Also, activation functions like ReLU discards the values that are smaller than 0. This loss of information can be compensated by using more number of channels (increased channel capacity).

Further, as we use the Inverted Residuals, we sqeeze the layers where the Skip connections are present. This hurts the performance of the network. So Linear Bottleneck layer makes sure that the last **Convolution** of the Residual block has a *Linear* output.

Check this article on [MobileNet V2](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5) for more details.

#### Accuracy vs Epochs graph for Training and Validation
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/graph_training_vs_validation_accuracy_over_epochs.png "Training vs Validation accuracy graph over number of Epochs")

#### 10 mis-classified image samples from each of the 4 classes
1. FlyingBirds
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/misclassified_samples/FlyingBirds_misclassified_samples.png "FlyingBirds Misclassified Samples")

2. LargeQuadCopters
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/misclassified_samples/LargeQuadCopters_misclassified_samples.png "LargeQuadCopters Misclassified Samples")

3. SmallQuadCopters
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/misclassified_samples/SmallQuadCopters_misclassified_samples.png "SmallQuadCopters Misclassified Samples")

4. WingedDrones
![MobileNetV2 Output](https://github.com/saneneurons/eva4p2/blob/master/Session-2/misclassified_samples/WingedDrones_misclassified_samples.png "WingedDrones Misclassified Samples")


### References:
- [Article explaining MobileNet V2 architecture](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)
- [MobileNet V2 Pre-trained weights from PyTorch Ecosystem](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
