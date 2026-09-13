# Financial Trend Prediction Model

This project implements a lightweight machine learning pipeline designed to predict daily asset price direction movements using historical technical indicators and a Random Forest classifier.

## Overview

The script generates synthetic price movement data using geometric Brownian motion, computes standard market technical indicators, and trains a supervised classification model to predict whether the next day's closing price will be higher than the current day. To prevent data leakage and look-ahead bias, it utilizes a time-series train-test split without shuffling.

## Features Engineered

* **Return**: Daily percentage change in closing prices.
* **Ratio**: Ratio of the 5-day simple moving average to the 20-day simple moving average.
* **Volatility**: Rolling 10-day standard deviation of returns.
* **Target**: Binary indicator representing whether the following day's closing price exceeds the current close.
