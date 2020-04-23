# Data Scientist Project

Project code for Udacity's Data Scientist Nanodegree program. The image classifier is built with PyTorch. Enable GPU to run this project. Training is done by using 102 different types of flowers, where there ~20 images per flower to train on. The trained classifier can be used to predict the type of a flower given the flower image. 

To train the model: 
```python train.py {input_data_dir} --arch alexnet --learning_rate 0.001 --hidden_units 200 --epochs 3 --gpu --dropout 0.5```

To predict the image from a trained model:
```python predict.py {input_img_dir} {checkpoint_path} --top_k 5 --gpu```

# Example
```============= Checkpoint Loaded =============
============= Predicting Image... =============
canterbury bells with a probability of 98.72%
black-eyed susan with a probability of 0.29%
desert-rose with a probability of 0.23%
lotus lotus with a probability of 0.18%
balloon flower with a probability of 0.10%
Prediction Done.```

