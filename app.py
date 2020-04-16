from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from  keras.models import Sequential, Model
from  keras.layers import Dense, Input
import numpy as np


def create_nn():
    # model = Sequential()
    # d0=Dense(70, input_dim=10000, activation='sigmoid')
    # model.add(d0)
    # d1=Dense(70, activation='sigmoid')
    # model.add(d1)
    # d2=Dense(1, activation='sigmoid')
    # model.add(d2)
    # return (model, d0, d1, d2)
    # Start defining the input tensor:
    inpTensor = Input((2,))

    # create the layers and pass them the input tensor to get the output tensor:
    hidden1Out = Dense(units=2)(inpTensor)
    hidden2Out = Dense(units=3)(hidden1Out)
    finalOut = Dense(units=1)(hidden2Out)

    # define the model's start and end points
    model = Model(inpTensor, finalOut)
    return (model, hidden1Out, hidden2Out, finalOut)
def fit_nn(X,Y):
    scores:list = None
    nn, d0, d1, d2=create_nn()
    es=EarlyStopping(monitor='val_accuracy')
    opt=SGD(lr=0.07)
    # opt='adam'
    nn.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    nn.fit(X, Y, epochs=10, validation_split=0.25)
    scores = nn.evaluate(X, Y)
    print("scores", scores)
    #, callbacks=[es])
    # vm_proc_print(b_c, locals(), globals())
    return (nn, d0, d1, d2)

def main():
   or_X=[[1, 1], [1, 0], [0, 1], [0, 0]]
   or_Y=[[1], [1], [1], [0]]
   or_X_np=np.array(or_X)
   or_Y_np=np.array(or_Y)
   fit_nn(or_X_np, or_Y_np)

if __name__ == '__main__':
    main()



