# Handwritten-math-expressions

All the code was tested on linux! 
Please use python3 and pytorch 0.3.1

## Database
We are using the CHROME 2016 database, please download it from the official website (need an account!): 

http://tc11.cvc.uab.es/datasets/ICFHR-CROHME-2016_1

## Generating More Images
We developped an algorithm capable of generating more data from the previous dataset. For running it, please use the following command,

```
python3 -m AugmData --directory PATH_TO_DATABASE --index-txt PATH_TO_GT --num-im 8000
```

It is important that the path of the database is linked to the folder where the inkml files are hosted ./TC11_package/CROHME2016_data/Task-2-Symbols/trainingSymbols

We are only interested in creating more images for the training set!!

You can vary the number of images per class you want by changing the number on the parameter.
 
## Generate PNG from inkml
Using the function developed by Harold Mouchère convert all the inkml files to png files using the following command,

```
python3 convertInkmlToImg.py PATH_TO_TRAIN_INKML 50 0
```

Please use this command to transform junk training, training, generated symbols training, validation training and test.

## Baseline
For running the baseline algorithm, please use the following command,

```
python3 -m Baseline --train PATH_TO_TRAIN_PNG --num-im 8000
```

Please give the complete path to the directory where the png images are stored. 

## Experiments with Neural Networks

For running any of the following neural networks please use the following command inside each available folder,

```
nohup python3 -u training_viejito.py --data PATH_TO_DATABASE --batch-size 102 --patience 3 >> NAME_OF_LOG.log &
```

All the options of the neural networks are part of the folders:

- /Alexnet
- /Concatnet
- /Stan
- /Concatnet_8_FC
- /Lenet
- /VGG13
- /Concatnet_FC
- /MNIST
- /Shallow
- /VGG6
- /Concatnet_K5
- /MNIST_K11


Enjoy!!!

