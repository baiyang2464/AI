#[CNN]Simple Convolutional Neural Network Applied to Face Emotion Recognition

+ Detector structure

  ​	getCsv.py			Make data sets with png pictures 

  ​	forward.py			Forward propagation

  ​	backward.py			Backward propagation

  ​	test.py				testing and get accuracy

  ​	face.csv				derived from getCsv.py

  ​	model				Directory of training model

  ​	classifier			Directory of face detector of openVC

  ​	jaffe				Directory of source pictures

+ Jaffe Database Download

  ​	[click here](http://www.kasrl.org/jaffe.html)

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

+ accurary

  %65(±10)

+ others

  1. when executing `python backward.py`, you may press `ctrl+c` to stop, then you can `python backward.py` again to continue training from last stop point.
  2. There are many Chinese explanations in the code.