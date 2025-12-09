# EKF-SLAM

Authors: Benjamin Weber, Danial Ruppin

## Project Structure
| Directory | Purpose |
|-----------|---------|
|```output/```|Target folder for outputted JSONL| 
|```slam_course/```|Contains the matlab code the algorithm implementation is based from|

| File | Purpose |
|------|---------|
|```dataloader.py```   |Purpose|
|```EKF_SLAM.ipynb```  |Demonstrates EKF-SLAM with graphs and metrics|
|```ekf_slam.py```     |Implements SLAM methods and required geometric methods|
|```eval_steps.md```   |Purpose|
|```main.py```         |Runs simulation and visualizer|
|```test_ekf_slam.py```|Test suite for methods in ```ekf_slam.py```|
|```visualizer.py```   |Purpose|

## Setup
Install the required python libraries by installing the requirements file:
```
$ pip install -r requirements.txt
``` 

## Use
The simulation can be run and visualized by running main.py:
```
$ python main.py
```
This will run the simulation and save the simulation history as ```output/slam_history.jsonl```  

The visualizer is then run on the saved simulation history and will open a window to display the simulation. 

## Output Format
The output is formatted as JSONL with each lines being a given time stamp of the simulation. The first line of the file corresponds to the Robot starting location and the ground truth locations of all the landmarks.

Example line with 1 instatiated landmark:
```
{"robot_position": [0.1, -2.0, 0.05], "map": [[-1.2013, 4.0429]]}
```