#@Kemal H Guddeta
# Implementation of LSTM using keras tensorflow backend
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#function that splits dataset into training and test
def split_data(data_set):
	train_size = int(len(data_set) * 0.80)
	test_size = len(data_set) - train_size
	train, test = data_set[0:train_size,:], data_set[train_size:len(data_set),:]
	return (train, test)

#function that builds the lstm model
def build_model(shape, neurons, dropout):
		model = keras.models.Sequential()
		model.add(keras.layers.LSTM(neurons[0], input_shape=(shape[0], shape[1]), return_sequences=False))
		model.add(keras.layers.Dropout(dropout))
		model.add(keras.layers.Dense(neurons[1],kernel_initializer="uniform",activation='linear'))
		model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
		model.summary()
		return model
#function that plost model loss functions
def model_history(model_history):
	history = model_history.history
	plt.plot(history['loss'], label = 'Training Loss', color = 'blue')
	plt.plot(history['val_loss'],label = 'Validation Loss', color='green')
	plt.legend(loc = 'upper right')
	plt.xlabel('Number of Epoch')
	plt.ylabel('Loss ')
	plt.show()
	
#Define function to get test score
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.4f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.4f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

#Make predictions
def make_prediction(train_x, test_x):
	trainPredict = model.predict(train_x)
	testPredict = model.predict(test_x)
	return (trainPredict,testPredict)

#invert predictions
def invert_predictions(train_predict, train_y,test_predict,test_y):
	trainPredict = scaler.inverse_transform(train_predict)
	trainY = scaler.inverse_transform([train_y])
	testPredict = scaler.inverse_transform(test_predict)
	testY = scaler.inverse_transform([test_y])
	return (trainPredict, trainY, testPredict,testY)

#Shidt predictions for plotting
def shift_predictions(data_set,train_predict,test_predict):
	trainPredictPlot = numpy.empty_like(data_set)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
	testPredictPlot = numpy.empty_like(data_set)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(train_predict)+(look_back*2)+1:len(data_set)-1, :] = test_predict
	return (trainPredictPlot,testPredictPlot)

#Plotting the actual vs predicted values
def plot_prediction(data_set, train_predict,test_predict):
	plt.plot(scaler.inverse_transform(data_set),'b', label = 'Actual Price')
	plt.plot(train_predict,'purple', label = 'Training Price')
	plt.plot(test_predict,'r', label = 'Predicted Price')
	plt.legend(loc = 'upper left')
	plt.xlabel('Time in Days')
	plt.ylabel('Apple Stock Price ')
	plt.show()

	
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('apple_close_price.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float64')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train, test = split_data(dataset)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#Set Configuration parametes
shape = [1, 1]
neurons = [256, 1]
dropout = 0.5
epochs = 50

#Build and Compile the model	
model = build_model(shape, neurons, dropout)

#Fit the model
model_return = model.fit(trainX,trainY,batch_size=5,epochs=epochs,validation_split=0.2,verbose=2)

#Plot Loss functions
model_history(model_return)	

#Get test scores print
model_score(model,trainX, trainY, testX, testY)

# make predictions
trainPredict,testPredict = make_prediction(trainX,testX)

# invert predictions
trainPredict, trainY,testPredict,testY = invert_predictions(trainPredict, trainY,testPredict,testY)
	
# shift train and test predictions for plotting
trainPredictPlot, testPredictPlot = shift_predictions(dataset,trainPredict,testPredict)

# plot actual vs predictions
plot_prediction(dataset,trainPredictPlot,testPredictPlot)