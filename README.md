# Handwriting-Generation-Using-Recurrent-Neural-Networks

This is an implementation of handwriting generation with use of recurrent neural networks (LSTM & GRU) using torch and python. Based on Alex Graves [paper](https://arxiv.org/abs/1308.0850).


## Installation

Clone the GitHub repository and install the dependencies.
* Install 
  * Anaconda (for creating and activating a separate environment)
  * pytest==3.2.1
  * numpy=1.13.3
  * matplotlib
  * tqdm==4.17.1
  * colorama==0.3.9
  * scikit-learn==0.19.1
  * pytorch==0.2.0 -c soumith

* Clone the repo and go to the directory 
```
$ git clone https://github.com/AizazSharif/Handwriting-Generation-Using-Recurrent-Neural-Networks.git
$ cd Handwriting-Generation-Using-Recurrent-Neural-Networks

```
* Run the **jupyter notebook** command in terminal and open notebooks/results.ipynb file for getting an overview of the whole project. 
* Go through each cell for a better understanding of the code and dataset.

## Training
The pretrained model is saved in models/ directory. Once you get the concept of **Conditional** and **Unconditional** Handwriting Generation you can train your own model by changing the setting in configurations.py.

For training the model use :
'''
python main.py --train_conditional
python main.py --train_unconditional
'''

## Validation
Validation can be simply done by running:

'''
python main.py --validate_conditional --conditional_model_path /Handwriting-Generation-Project/models/conditional.pt
python main.py --validate_unconditional --unconditional_model_path /Handwriting-Generation-Project/models/unconditional.pt

'''

## Credit 

This project is a part of Deep Learning related task and all credit goes to [Lyrebird](https://lyrebird.ai/) for introducing me to this incredible work.





