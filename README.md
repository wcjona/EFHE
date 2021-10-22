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
