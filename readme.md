# Transfer Learning
A simple example of transfer learning using the inception v3

>The follwing uses transfer learning for classifying hand sign numbers, digits from 0 to 9

In the following repo I've implemented a basic example of transfer learning using the inception v3 model. The approach is fairly simple and can be understood as:
<br>
<pre>
1. Load your Datasets (train, test, validate)
            |
2. Load the Inception Model
            |
3. Create additional layers you need
            |
4. Train the model
            |
5. Evaluate the model.
</pre>

The amount of images in each dataset:
<pre>
Training: 994
Validation: 568
Test: 500
</pre>

> To run the program as it is
1. Download or clone the repo.
2. Go in to the repo folder and then run task.py using
` python task.py`

>**NOTE:** The program has two different training approach one with only the training dataset and with training + validation dataset and to use them, just keep one of them commented and the uncomment the other or the model may end up getting trained twice.

>***NOTE:*** Make sure you have tensorflow 2.x, numpy and matplotlib installed before running the code.
<br>
<pre>
To install:
    tensorflow -> pip install tensorflow
    numpy      -> pip install numpy
    matpltlib  -> pip install matplotlib
</pre>
> If you have an cuda enabled gpu you can install tensorflow gpu
<pre>
    tensorflow GPU -> pip install tensorflow-gpu
</pre>

>***References and sources:***

>**dataset**: <br>https://github.com/ardamavi/Sign-Language-Digits-Dataset <br>

>**Others**:<br>
https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/ <br>
https://www.tensorflow.org/api_docs/python/tf/keras

