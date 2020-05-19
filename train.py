# import os
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from wandb.keras import WandbCallback

def main():

    print('Carga de librerias correcta')

    hyperparametres={"epochs":     235,
                     "batch_size": 12,
                     "Dropout":    0.3,
                     "optimizer": 'Adam'
                    }

    wandb.init(config=hyperparametres, project='Iris_with_ANN')
    config = wandb.config

    print('Se va a cargar el archivo')

    data = pd.read_csv('C:/Users/Kevin Pulido/iris.csv')

    print('Importacion de archivo correcta')

    LabelChanger = LabelEncoder()
    LabelChanger.fit(data['Species'])
    data['Clases'] = LabelChanger.transform(data['Species'])

    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = data['Clases']

    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size = 0.15,
                                                        stratify = Y)

    # Mean Normalization (Standarize the features to follow the normal distribution, to obtain a faster & better classifier)
    sc = StandardScaler()

    X_train_array = sc.fit_transform(X_train.values) #calculate μ & σ(fit) and apply the transformation(transform)

    # Assign the scaled data to a DataFrame & use the index and columns arguments to keep your original indices and column names:
    X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)

    # Center test data. Use the μ & σ computed (fitted) on training data
    X_test_array = sc.transform(X_test.values)
    X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

    # Initializing Neural Network
    model = Sequential(name='ANN')

    model.add(Dense(units=12, activation='selu', kernel_initializer = 'lecun_normal', input_dim=4,))
    model.add(Dropout(config.Dropout))
    model.add(Dense(units=12, activation='selu', kernel_initializer = 'lecun_normal'))

    model.add(Dense(units=3, activation='softmax', kernel_initializer = 'GlorotNormal'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=config.optimizer,
                  metrics=['accuracy']
                 )

    Historia = model.fit(X_train.values,
                         Y_train.values,
                         epochs = config.epochs,
                         batch_size = config.batch_size,
                         validation_data = (X_test.values, Y_test.values),
                         callbacks=[WandbCallback()]
                        )

    lss, acc = model.evaluate(X_train.values,
                              Y_train.values,
                              batch_size=config.batch_size,
                              verbose=2)

    # Log metrics inside your training loop
    metrics = {'accuracy': acc, 'loss': lss}
    wandb.log(metrics)

    print('Model accuracy:', acc)

    # the prediction results
    predicciones = model.predict_classes(X_test.values, batch_size=config.batch_size, verbose=1)

    real_acc = np.sum(predicciones == Y_test)/23.0 * 100

    print("Test Accuracy : " + str(real_acc) + '%')

if __name__ == '__main__':
   main()
