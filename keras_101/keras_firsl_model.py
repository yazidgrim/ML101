from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#read csv file
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

#X attributes array - y results array (1 has diabetes - 0 don't have diabetes)
X = dataset[:, 0:8]
y = dataset[:, 8]

#define keras model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

#evaluate the model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

