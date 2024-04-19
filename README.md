# EF Hand Evaluator (EFHE) by iGEM Calgary
Welcome to EFHE, an program able to predict if a protein sequence contains EF hands or not. It will also differentiate confirmed and potential EF hands. 

## Installation
The following packages must be installed prior to running EFHE
```
pip install -U pandas 
pip install -U numpy  
pip install -U cPickle
pip install -U sklearn
pip install -U tensorflow 
pip install -U keras
pip install -U sklearn
```

## Usage
Start by cloning the reposititory by using Console or the Github Application. Once installed, make sure the dataset files are accessible. 
The ```EFhandConverter.ipynb``` converts all the reviewed and unreviewed protein sequences into their seperate EF hands, this program is also responsible for developing a set of false EF hands. The two datasets were combined with a label, ```1``` representing an EF hand, and ```0``` representing a non-EF hand. The resulting dataset is named ```efHandData_neg_pos_interleukins.csv``` and can be found in the ```Datasets``` folder.  
Next, using the ```efHandData_neg_pos_interleukins.csv``` dataset, the ```EFhand_CNN+LSTM.ipynb``` script is run to develop Bi-CNN + LSTM models for predicting EF hands. We manipulated the ```nb_epoch```  value to evaluate which model would give the best accuracy. The resulting models are listed in the ```Results``` folder. 
The final ```EFHE.ipynb``` script is used to evaluate protein sequences. Simply input a protein sequence into the ```sequence``` value, and ensure the desired epoch model is correctly imported. The result is value containing the number of confirmed ef hands, and the number of potential ef hands. It also includes an prediction value, as well as the sequence and the location it occurs within the protein sequence. A sample using lanmodulin's sequence is provided. 

----------------------------------------------------------------------------------------------------
https://2021.igem.org/Team:Calgary/EFHE

## Reason
While characterizing the fusion proteins created for the biosensor, we discovered that some of the binding pockets from lanmodulin exhibited weak binding affinity when compared to the others. Additionally, we found a fourth pseudo binding pocket with potential to be mutated into a viable pocket, which motivated the development of an EF hand optimizing and evaluating workflow. EF hands are structural motifs that play a critical role in the binding functionality of lanmodulin and other proteins. Their sequences do not have a rigid definition but are rather composed of negatively electrostatic binding regions formed by aspartic and glutamic amino acids in a semi-sphere. Due to the variety of intricate motifs found with EF hands and the absence of tools to recognize and evaluate them, we decided to develop our own program to identify and classify sequences based on their inclusion of the EF hand motif.
Introducing our EF Hand Evaluator (EFHE), a program that uses a deep learning neural network to accurately predict the number of EF hands, their specific location, and its effectiveness based on protein sequence alone. EFHE achieves this using two Convolutional Neural Network (Bi-CNN) layers with a Long Short Term Memory (LSTM) Model [1].

## Introduction to Deep Learning
Deep learning is a subset of machine learning which employs the use of neural networks to simulate the processing that the human brain goes through when learning. On a fundamental level, deep learning models are able to take in labeled training data, and learn the patterns that enable the inputted data to result in a specific label output [2]. For instance, deep learning neural networks are often used for entity recognition within images such as cars, stop signs, and people. In theory, this involves identifying trends within pixel values like orientation, shape, and size. Additionally, deep learning neural networks can also be used for text processing, like predicting the next word in a sentence given the content of the first half. For our applications, we plan on using neural networks for one-dimensional protein amino acid sequences, and label each sample based on whether or not an EF hand is present.

## Our Model
Within deep learning, Bi-CNN LSTM models are specifically designed for sequence prediction problems similar to ours [3]. These types of problems are specifically known as time-series problems in which the order that things occur matters. Predicting the weather is a perfect example of a time-series problem as it depends on a multitude of variables resulting in a specific temperature which may vary from hour to hour. Using deep learning, we can identify trends and develop a forecast. Similarly, when looking at a part of a protein, the type of amino acid will vary depending on the position in the sequence you are looking at. Therefore, by using the position in the sequence as our “time” variable we are able to introduce current deep learning methods to our problem.

![image](https://github.com/wcjona/EFHE/assets/46095400/6f784e57-d892-4942-8d1a-8e75e39d2fc5)

*Figure 1. The Structure and specific layers of our deep learning model*

### Convolutional Neural Network
A Convolutional Neural Network is a network made up of neurons which have their own trained weighting and bias. These neurons will adjust to the given input overtime to improve on itself. A layer of neurons then creates a matrix (also known as a filter) and is applied to the input data. This is also known as convolution, since the output of this step is then fed into another filter and continues until the desired number of filters is reached. As the training data goes deeper into the neural network, the filters will identify more specific motifs compared to the first few layers. The output is then moved into a pooling layer which uses downsampling to find the most significant features and generalize the data [4]. Finally, a second CNN (Bi-CNN) layer was implemented into our system when we noticed it marginally increased the accuracy of the model. This output is then fed into a long short-term memory layer.

### Long Short-Term Memory Model
LSTM layers contribute to better results as it considers the previous values when predicting what the next value will be. For example, if the weather was hot and sunny for the last 3 hours, the chances that it will be hot and sunny an hour later would be high. Similarly, given the same information, the chances that it will snow would be considerably low. LSTMs introduce this idea through memory blocks, a complex processing unit made up of memory cells. Each memory cell contains three multiplicative gates: an input gate, forget gate, and output gate. If values match the criteria of an input gate, it will be used and transmitted to an output gate. During training, if the output consistently leads to an incorrect answer, the forget gate is used to reset the state of the memory cell. This allows memory blocks to remember and connect previous information to the present, and remove pieces of memory that lead to poorer predictions [4]. By including this layer after the Bi-CNN layers it is able to leverage the motifs identified by the convolutional layers. This means that the layer is processing and handling latent data, non obvious trends, not just the raw sequences. The final layer, known as the fully connected layer, gives a probability distribution over each feature to produce a final output.

## Implementation
Using Keras, a neural network library built on tensorflow, we were able to use predefined functions for the embedding layer, 1D convolutional layers, pooling layers and the LSTM layer [5]. We simply had to format the training data so that it is readable for the neural network, and tune the required hyperparameters to our liking. Tuned hyperparameter values can be found on our repository.

### Embedding Layer
First the data is fed into the embedding layer, which requires the input data to be integer encoded. This means that each amino acid is represented by a unique integer. The embedding layer then converts each word into a fixed length vector which can be then used by the CNN [6]. One of the parameters of the embedding layer is known as the maximum number of features. This is the largest number of unique integers a position can hold. In addition to the 21 different proteinogenic amino acids, two more spots were needed for any unknown characters and empty positions. As such we put 23 as the maximum number of features. A function was also created to convert amino acid sequences into their respective unique integers [1].


The next two parameters for this layer are the maximum length and the embedding size. Because each EF Hand has a length of 12 amino acids, the fixed length of each vector will be 12. On the other hand, the embedding size essentially equates to how compressed we want the vector information to be. A larger dimensionality (or embedding size) typically results in more lexical detail but if used on too small of a dataset, it may result in overfitting. The general rule of thumb is a value between 50 to 300 for less computationally expensive systems. Therefore, in order to find the value that would give the best results, we tested our system with 64, 128 and 256 embedding dimensions. Using this method the following results were gathered.

![image](https://github.com/wcjona/EFHE/assets/46095400/508e33c6-02c0-47e7-b617-be6a1b9509d5)

*Figure 2. Where “std” is the standard model, and “cur” is the current model and “m” is the mean squared error, while “n” is the number of features*

As a result from the table above, 128 was chosen as our embedding size due to its increased accuracy over the others.

### 1D Convolutional Layer
The 1D convolutional layer requires the number of filters and the pool length. This is the number of neurons and size of the output from the pooling layer respectively [7]. Based on literature review and the computational resource constraints, we decided to choose 10 filters and a pool length of 2 [1].

## Training Function
Finally the batch size and number of epochs are required for the training function. The batch size is the number of samples taken from the training set at a time, this is mainly dependent on the size of our dataset, therefore we decided to choose 128. On the other hand, an epoch is when the entire dataset is processed through the neural network forwards and backwards [8]. If the number of epochs is too big, the model will be overfit leading to more missed ef hands. Although if too small, the model will be underfit resulting in more false positives. Because of this, we decided to test multiple epochs to find a balance between under and overfit models. The following table was developed to find the optimal number of epochs. Ultimately, we found 150 to be the balance with the best results.

## Dataset
Using the process mentioned, we now needed a way to train our data using verified ef hands. Lucky, uniprot provides comprehensive, high quality proteins with functional information [9]. Additionally, the results can be customized to show specific details of each protein and then can be converted into a csv. Using this, we simply searched up “EF hands” in the database and exported a list of proteins containing ef hands and specific regions that they exist. A script was then created to extract the EF hand sequences from each protein given their location in the sequence. Using interleukins, a group of naturally occurring proteins that do not contain EF hands, we created a list of false EF hands containing EF hand motifs such as 12 length sequences and recurring aspartic and glutamic amino acids. True EF hands were then given a value of 1 while the false ef hands were given a value of 0. The result was over 2 thousand of both resulting in 4 thousand samples.

## Finished Model
Finally after testing our model with different numbers of epochs and embedding sizes, our best model had a 97.91% accuracy on the test dataset consisting of 800 samples. Considering the rigorousness of our dataset we see this as a huge success. The model can finally be used to evaluate if a part of a protein is an ef hand, additionally because the output is a value between 0 and 1, it also provides the ability to detect potential or weak EF hands. Through trial and error, a threshold was then created to classify whether an sequence was confirmed to be an EF hand, had potential of being an ef hand or was neither.

### Formatting for the input
Because the goal was to create a program that finds and evaluates EF hands within a protein sequence, we now needed to format the input so that it would work with the model we developed. We noticed that the majority of the ef hands start with a D (Aspartic Acid), and in some cases an E (Glutamic Acid) or A (Aline). Because of this, we used this motif to cut 12 length sequences from a protein if they contain each of these letters. Another issue we encountered is that if only part of a true EF hand is fed into the model, it may be recognized as a potential EF hand. This resulted in multiple potential EF hands overlapping with the true EF hand. To account for this, the software would remove a potential sequence,if a sequence with a higher score overlapped it. We based this on the position the sequences were located in the protein to prevent the edge case in which there are identical sequences within the same protein.

## Results
Ultimately we used this program not only to evaluate the effectiveness of the binding pockets in our fusion proteins, but this model served to be extremely useful when trying to optimize lanmodulin’s binding pockets. Ultimately, this program served as a contribution to future teams that wish to evaluate and locate EF hands within their novel protein. Along with uncovering EF hand motifs, this model can be used as a template for other teams looking to develop their own sequence based motif discovery models. If you would like to access EFHEs code it is available on the iGEM GitHub here.

### Future Steps
In the future we would like to compare a variety and combination of different deep learning methods in order to optimize the model. Additionally, we see huge potential in applying deep learning to more synthetic biology related software due to its ease of access and accuracy.

## References
Qu Y-H, Yu H, Gong X-J, Xu J-H, Lee H-S. On the prediction of DNA-binding proteins only from primary sequences: A deep learning approach. PLOS ONE. 2017 Dec 29 [accessed 2021 Oct 21]. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0188129#pone-0188129-g001

Brownlee J. What is deep learning? Machine Learning Mastery. 2020 Aug 14 [accessed 2021 Oct 21]. https://machinelearningmastery.com/what-is-deep-learning/

Wu C, Wu F, Chen Y, Wu sixing, Yuan Z, Huang Y. Neural metaphor detecting with CNN-LSTM model. Tsinghua University. 2018 Jun 6 [accessed 2021 Oct 21]. https://aclanthology.org/W18-0913.pdf

Sethuraman P. A comparison of DNN, CNN and LSTM using TF/Keras. Medium. 2020 Sep 24 [accessed 2021 Oct 21]. https://towardsdatascience.com/a-comparison-of-dnn-cnn-and-lstm-using-tf-keras-2191f8c77bbe

Team K. Keras Documentation: Keras API reference. Keras. [accessed 2021 Oct 21]. https://keras.io/api/

Team K. Keras documentation: Embedding layer. Keras. [accessed 2021 Oct 21]. https://keras.io/api/layers/core_layers/embedding/

Brownlee J. A gentle introduction to pooling layers for Convolutional Neural Networks. Machine Learning Mastery. 2019 Jul 5 [accessed 2021 Oct 21]. https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

Team K. Keras Documentation: Model training apis. Keras. [accessed 2021 Oct 21]. https://keras.io/api/models/model_training_apis/

UniProt ConsortiumEuropean Bioinformatics InstituteProtein Information ResourceSIB Swiss Institute of Bioinformatics. Uniprot Consortium. UniProt ConsortiumEuropean Bioinformatics InstituteProtein Information ResourceSIB Swiss Institute of Bioinformatics. [accessed 2021 Oct 21]. https://www.uniprot.org/
