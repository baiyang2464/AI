# [CNN]Simple Convolutional Neural Network Applied to Face Emotion Recognition

+ File structure

  **emotion**

  ​	-[getCsv.py](./getCsv.py)&emsp;(Make data sets with png pictures )

  ​	-[forward.py](./forward.py)&emsp;(Forward propagation)

  ​	-[backward.py](./backward.py)&emsp;(Backward propagation)

  ​	-[test.py](./test.py)&emsp;(testing and get accuracy)

  ​	-face.csv&emsp;(derived from getCsv.py)

  ​	-**model**&emsp; (Directory of training model)

  ​	-**classifier**&emsp;(Directory of face detector of openVC,)

  ​	-**jaffe**&emsp;(Directory of source pictures,you should download it first)

+ Jaffe Database 

  &emsp;[click here to download database](http://www.kasrl.org/jaffe.html) &ensp;

  &emsp; The database contains 213 (resolution of each image: 256 pixels × 256 pixels) Japanese women's face, each image has an original expression definition. There are 10 people in the expression library, each with 7 expressions (neutral face, happy, sad, surprised, angry, disgusted, fear,use interger number of 0-6 to express)

  &emsp;I used the first 200 images for model training and tested with the last 13 images.

+ Environment and main Tools

  ​	Ubuntu 16.04

  ​	python

  ​	tensorflow

  ​	openCV

+ Execution steps

  + `python getCsv.py` to get `face.csv`
  + `python forward.py `
  + `python backward.py`
  + `python test.py`

+ accuracy

  %65(±10)

+ others

  1. when executing `python backward.py`, you may press `ctrl+c` to stop, then you can `python backward.py` again to continue training from last stop point.
  2. There are many Chinese explanations in the code.
