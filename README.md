# Description

This repository contains the code for an app developed with FastApi and deployed on Heroku with CI / CD.

A simple machine learning model is trained on us census data (https://archive.ics.uci.edu/dataset/20/census+income) and is used to predict salary. When data is sent to the endpoint a salary prediction is returned (< or > then 50k) based on socio economic input data.

The project is part of the Udacity MLOps Nanodegree.

# How to

The app can be accessed via https://cg-census-pred-app-prod-e1af23d4a843.herokuapp.com

You can test the endpoint with the script live_api_testing.py where you will get the GET response as well as the POST.