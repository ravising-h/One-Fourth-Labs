from keras.layers import Input,Conv2D,Dense, Dropout, BatchNormalization, MaxPool2D, Activation, Flatten, AvgPool2D,GlobalMaxPooling2D # KERAS LAYERS
from keras.layers import  BatchNormalization as btn # BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential  #model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam


def _models_(nets = 1,opt = "adam",weights = None):
	
	model = Sequential()

	model.add(Conv2D(128, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size = 5, strides=2, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.44))

	model.add(Conv2D(1024, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dropout(0.44))
	#mode.add(Dense(64,activation = 'relu'))
	model.add(Dense(47, activation='softmax'))


	# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
	model.compile(optimizer='RMSprop', loss="categorical_crossentropy", metrics=["accuracy"])
	print(model.summary())
	return model