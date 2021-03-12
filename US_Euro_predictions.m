% We'll first connect to the FRED data server using the url:
url = 'https://fred.stlouisfed.org/';
c = fred(url);
%% 
% Adjust the display data format for currency.
format bank
%% 
% Retrieve historical data for the US/EUR exchange rate
% series.
series = 'DEXUSEU'; 
%% 
% Fetch 2 years of data from Sept 1, 2017 through Sept 1, 2019.
startdate = '09/01/2017';
enddate = '09/01/2019';
d_input = fetch(c,series,startdate,enddate)
Input= (d_input.Data)
% replace missing data (N/A) values with data at previous timestep
clean_input= fillmissing(Input,'previous');

DATA=clean_input
% converting the dates from datenum format to datetime format
TimeDT = datetime(DATA(:,1), 'ConvertFrom', 'datenum', 'Format', 'yyyy-MM-dd');
% visualizing our data
figure
plot(TimeDT,DATA(:,2))
xlabel("Date")
ylabel("Exchange rate")
title("US/Euro Exchange Rate")

%% 
% Partition the training and test data. Train on the first 90% of the sequence 
% and test on the last 10%.

% We'll make an array pricedata which only has the prices (exchange rates).
% We also transpose this, to make a 1xN array
pricedata= (DATA(:,2))'
% splitting the data
nTimeStepsTrain = floor(0.9*numel(pricedata));

trainData = pricedata(1:nTimeStepsTrain+1);
testData = pricedata(nTimeStepsTrain+1:end);

%% Standardize Data
% For a better fit and to prevent the training from diverging, we'll standardize 
% the training data

mu = mean(trainData);
sig = std(trainData);

trainDataStandardized = (trainData - mu) / sig;

%% Prepare Predictors and Responses
% For forecasting values of future timesteps we describe the responses to
% be the training sequences with values moved by one time step
X_train = trainDataStandardized(1:end-1);
Y_train = trainDataStandardized(2:end);
%% *Here, we'll define the LSTM Network Architecture*
% Create an LSTM regression network. Specify the LSTM layer to have 200 hidden 
% units.

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%%

% Specifying the training options. Solver is set to 'adam'. Gradient
% threshold is set to 1 (to prevent gradients from exploding). Initial
% learn rate is 0.005 which drops after 125 iterations by multiplying with a factor of 0.2

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train LSTM Network
% We now train the network using the specified training options by using trainNetwork

network = trainNetwork(X_train,Y_train,layers,options);
%% Forecasting Future Time Steps
% We'll use the predictAndUpdateState function to estimate values of
% multiple time steps in the future. This function updates network state at each prediction. For each prediction we'll use the
% previous prediction as the input to the function

% Standardizing the test data using the same parameters as with the
% training data.

testDataStandardized = (testData - mu) / sig;
X_test = testDataStandardized(1:end-1);
%% 
% To initialize the network state we first predict on the training data |X_train|. 
network = predictAndUpdateState(network,X_train);
% Make the first prediction using the last time step of the training response |Y_train(end)|
[network,Y_pred] = predictAndUpdateState(network,Y_train(end));

% Loop over remaining predictions and input the previous prediction 
% to |predictAndUpdateState|.
nTimeStepsTest = numel(X_test);
for i = 2:nTimeStepsTest
    [network,Y_pred(:,i)] = predictAndUpdateState(network,Y_pred(:,i-1),'ExecutionEnvironment','cpu');
end

%% 
% Unstandardizing the predictions using the paramteres calculated from before
Y_pred = sig*Y_pred + mu;
%% 
% The training progress plot reports the root-mean-square error (RMSE) calculated 
% from the standardized data. RMSE will be calculated from the unstandardized 
% predictions.

Y_test = testData(2:end);
rmse = sqrt(mean((Y_pred-Y_test).^2))
%% 
% Plot the training time series with the forecasted values.

figure
plot(trainData(1:end-1))
hold on
idx = nTimeStepsTrain:(nTimeStepsTrain+nTimeStepsTest);
plot(idx,[pricedata(nTimeStepsTrain) Y_pred],'.-')
hold off
xlabel("Date")
ylabel("Exchange rate")
title("Forecast")
legend(["Observed" "Forecast"],'Location','northeast')
%% 
% Compare the forecasted values with the test data.

figure
subplot(2,1,1)
plot(Y_test)
hold on
plot(Y_pred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Exchange rate")
title("Forecast")

subplot(2,1,2)
stem(Y_pred - Y_test)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)