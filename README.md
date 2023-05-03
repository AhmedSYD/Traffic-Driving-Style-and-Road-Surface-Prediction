# Traffic-Driving-Style-and-Road-Surface-Prediction

ENGO 645 - Spatial Databases & DataMining

## Overview:
This project is the final project for the course (ENGO 645 - Spatial Databases & DataMining) that was taught at the University of Calgary. In this project, I  will investigate all possible solutions to estimate the parameters of road conditions, traffic conditions, and driving style of the vehicle to be used in the near future to check if driving in a road is safe or not. In this project, I will present one of the optimal solutions to predict the environmental condition, traffic congestion situation and road surface, and driving style, peace or aggressive style. In the near future, these predictions can be utilized to estimate if the car will make an accident or not. The three prediction variables will be estimated depending on car features such as engine load and coolant and other speeding and acceleration features by utilizing the [**Kaggle dataset**](https://www.kaggle.com/datasets/gloseto/traffic-driving-style-road-surface-condition). This dataset is collected from two types of cars, namely Peugeot 207 1.4 HDi (70 CV) and Opel Corsa 1.3 HDi (95 CV). I utilized different types of data mining algorithms and data preprocessing for achieving the project's goal. Then, I evaluated each method and picked the best one that achieve the highest prediction accuracy as indicated in the document below.


## System requirement:
- Any platform you like such as Windows, Linux, and so on. 
- python 3.6 or higher

## Libraries required to install:
- matplotlib
- numpy
- pandas
- scikit_learn
- scipy
- seaborn
You can find all of these libraries in the `requirements.txt` and install all of them by running this command `pip3 install -r requirements.txt` in the terminal window.

## How to run the project:
* After installing all libraries required in your environment, run `project_processing.py` in any IDE you like for applying all required data cleaning and processing on the dataset files that are located in the dataset folder. You will notice after running the code that the folder (`preprocess_dataset`) is created that contains the data after preprocessing. 
* Then, run the `project_algorithm.py` file to start the prediction of the attributes of environment conditions, traffic congestion situation, and road surface by utilizing multiple machine learning models. 

## Documentation:
- You can find the detailed document of this project in this [**link**](https://drive.google.com/file/d/1g6GYU7ixhuv_FSUs-cSYPFKAJVHIc3cF/view?usp=sharing) and the presentation in this [**link**](https://docs.google.com/presentation/d/1P5E_CNhrBklg7KPJv7XU56drGKPVfJXJ/edit?usp=share_link&ouid=114144978379823643027&rtpof=true&sd=true).
