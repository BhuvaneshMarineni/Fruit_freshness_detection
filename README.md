# Fruit_freshness_detection_computerVvision
Fruit fresheness detection

## Bhuvanesh Marineni 

## Introduction
Detecting fruit freshness is an essential responsibility in the food sector, as it ensures that customers obtain high-quality and safe goods. One method for achieving this objective is using computer vision methods to evaluate photographs of fruits and identify symptoms of rot or maturity. OpenCV is a powerful computer vision toolkit that may be used to construct Python-based systems for fruit freshness detection. OpenCV offers a variety of image processing capabilities, such as image filtering, edge detection, and object identification, which may be used to fruit photos to extract valuable characteristics and categorize fruits according to their freshness. 

### Aim and objectives

## Aim

The aim of the project is to develop a fruit freshness detection system using OpenCV and Python that can determine the freshness of fruits with precision.
Objectives
The objectives of the project have been provided below: 
To preprocess the data set using methods like feature extraction, resize, crop, and normalization.
To categorize fruits depending on their freshness state, train and assess machine learning models using the retrieved attributes.
To fine-tune the model's hyperparameters to improve its performance and accuracy in detecting fruit freshness.
To use OpenCV images processing methods like filtering, edge detection, and object identification.

## Methods

This fruit freshness detection project's approaches include gathering a collection of fruit photos, preprocessing them, extracting key features using OpenCV, training different machine learning models, fine-tuning its hyperparameters, and assessing the system's performance (Sharma et al. 2022). In order to prepare the photos for feature extraction, they are resized, cropped, and normalized. Using OpenCV image processing methods like filtering, edge detection, and object identification, relevant characteristics are retrieved from the preprocessed fruit pictures. The models are fine-tuned to maximize their performance and enhance their precision in detecting fruit freshness (Le and Mohd, 2022). The fruit freshness detection system is developed as a Python program that accepts photos of fruits as input and returns the freshness state of those fruits as output.

![image](https://user-images.githubusercontent.com/122952070/236652039-36279166-59fa-497f-9609-9da336a5da26.png)


This code is used to get the images from a directory and put them into an array. The directory containing the photographs is saved as a path, which is subsequently used to access the images. A for loop then reads in the photos and resizes them to 224 by 224 pixels. The photos are transformed from BGR to RGB color coding and stored in an array, which is then returned and saved as a numpy array. The 'tqdm' module is used to generate a progress bar so the user can monitor the number of photographs put into the array.

## Figure 2: Visualizing the images

![image](https://user-images.githubusercontent.com/122952070/236652119-ab3580ab-d5b0-4a40-b4cc-3130ad56caa5.png)

This code displays the photos included in the ndarray "arr img" The code begins by declaring the function img_disp, which accepts a single input "X," followed by an if statement. If the initial dimension of the array is 36, a 6x6 plot is generated using the plot function of the matplotlib module with a 40X60 figure size. Next, the loop is begun to plot each picture in the array on the corresponding plot and set the plot title to the name of the fruit saved in list_fruits. The plot is then presented. If the first dimension of the array is not 36, the error message "Cannot plot" is displayed. The function with the array of arguments is then invoked at the conclusion of the code.

## Figure 3: Generating the data of the images

<img width="468" alt="image" src="https://user-images.githubusercontent.com/122952070/236652128-348a05b1-781a-451a-bfd0-979b1a0385f0.png">


This code creates the datagen and datagen1 ImageDataGenerator objects. Many factors in datagen govern “picture alteration, including rotation range, width, height, shear and zoom shifts, as well as horizontal and vertical flipping”. The rescale parameter rescales the pixel intensity values, while the preprocessing function parameter applies picture preprocessing functions such as VGG16 or Inception. The simpler datagen1 has just a rescale parameter and a preprocessing function, thus the photos can only be rescaled and preprocessed and can not be further modified.

## Figure 4: Classifying the images 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/122952070/236652134-91b5b086-c7cf-4263-b731-c227c81df836.png">

This code generates data for a Deep Learning (DL) model where the train gen variable is initialized by the Keras method datagen.flow from directory(). It automates the generation of batches of enhanced data from the directory containing the train pictures. The target size is then set to [224,224], and the color mode is set to RGB. The classes must be configured as a list of categorical data, and the batch size must be set to 32. Similarly, datagen.flow from directory () likewise initializes the val gen variable with the same goal size, color mode, class mode, and batch size. 

## Figure 5: Accuracy of the model

<img width="468" alt="image" src="https://user-images.githubusercontent.com/122952070/236652152-7cb621d5-2fc7-41c3-9e8f-4bbf7af8ee43.png">


This code illustrates the Model Accuracy notion using a graph created from the vgg history object's data. The graph is drawn using the 'accuracy' and 'val accuracy' keys of the vgg history object. Model Accuracy is the title of the graph, with Epochs and Accuracy labeling the x- and y-axes, respectively. In addition, a legend is provided showing whether the displayed line corresponds to 'Acc' or 'Val' data. 

## Figure 6: Model Loss

<img width="468" alt="image" src="https://user-images.githubusercontent.com/122952070/236652158-ff785c9a-377c-4e0c-b918-c62047ae4461.png">

The code demonstrated in the above image represents the loss of a VGG model over time. The lines of code use the function plt.plot to plot the training loss and validation loss histories. The two losses are labeled after the xlabel and ylabel, which indicate what is being plotted on the graph, are explained. Using the command legend, the title of the graph and the names of the several plots are specified. 

## Conclusion


In conclusion, a fruit freshness detection system capable of reliably classifying fruits depending on their freshness condition has been constructed using OpenCV and Python. Using OpenCV image processing tools, the important characteristics from the dataset have been extracted, and machine learning models have been trained to categorize fruits depending on their freshness condition. The hyperparameters of the model have been fine-tuned to maximize its performance and improve its fruit freshness detection accuracy.

## Link for the dataset is given below

Dataset link: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification


## References

https://www.mdpi.com/2223-7747/12/4/790/pdf

https://www.mdpi.com/1424-8220/22/21/8192

https://www.irjmets.com/uploadedfiles/paper/volume2/issue_6_june_2020/1554/1628083042.pdf

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017387/

https://www.irjet.net/archives/V7/i5/IRJET-V7I51254.pdf

https://www.researchgate.net/publication/368377088

https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification


