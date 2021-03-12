# US-EUR_ExchangeRate
Prediction of USD/EUR exchange rate using data from Sept 2017 to Sept 2019
We used a 90/10 train/test split and obtained a RMSE of 0.012605
The predictions using a Long-Short-Term-Memory neural network in MATLAB is visualized below.
<img width="381" alt="FXforecast" src="https://user-images.githubusercontent.com/29689235/110972858-64689880-8354-11eb-891b-5bef3bf64910.png">

We also predict the exchange rate in Python which provided us with more accurate results with the same number of iterations (250). We used a 80/20 train/test split
<img width="541" alt="python_predictions" src="https://user-images.githubusercontent.com/29689235/110995986-23cc4780-8373-11eb-8325-90d4c515e709.PNG">

The True vs Predicted prices with Python are shown below

<img width="89" alt="python_predictions2" src="https://user-images.githubusercontent.com/29689235/110996053-41011600-8373-11eb-885d-0ac46dfe23f3.PNG">

Using Python to predict prices gave us an RMSE of 0.009177 which was better than the RMSE obtained with the MATLAB model - 0.012605
