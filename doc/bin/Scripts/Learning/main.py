from tensorflow import keras
from keras.layers import Dense,Activation
from keras.models import Sequential, load_model
from keras import optimizers
from keras import backend as K

import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np


class Methods:
    def Power(self,x,y): 
        for i, value in enumerate(x):
            x[i] = float(math.log(value))
            
        for i, value in enumerate(y):
            y[i] = float(math.log(value))
            
        values = np.polyfit(x,y,1)
        c = math.exp(values[1])
        b = values[0]
        return c,b   
    
    def Pattern(self,data):
        _pattern = ""
        for index,value in enumerate(data):
            _pattern +=str(value)+":"      
        return _pattern[0:len(_pattern)-1]
    
    
class model_DNN(Methods):
    
    code = None
    dataset = None
    weight = None
    distance = 1
    optimizers,losses,callbacks = None,None,None
    X_train,X_test,y_train,y_test = None,None,None,None
    
    
    def _load(self,code,dataset,weight,distance):
        self.code=code
        self.dataset = dataset
        self.weight=weight
        self.distance = distance
        
        sgd = optimizers.SGD(lr=0.01)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        RMSprop = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        Adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        Adadelta =optimizers.Adadelta(lr=0.1, rho=0.95, decay=0.0)
        Adamax = optimizers.Adamax(lr=0.02, beta_1=0.9, beta_2=0.999,  decay=0.0)#02
        Nadam = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        Adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0)
        sgd2 = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        
        callback = keras.callbacks.Callback()
        history = keras.callbacks.History()
        
        self.optimizers = [sgd,adam,RMSprop,Adagrad,Adadelta,Adamax,Nadam,Adam,sgd2]
        self.losses = ['mse','mean_absolute_error','mean_absolute_percentage_error','cosine_proximity']
        self.callbacks  =[callback,history]
    
    def _set(self):
        model = Sequential()
        model.add(Dense(units=75, input_dim=1))
        model.add(Activation('relu'))
        model.add(Dense(units=24))
        model.add(Activation('relu'))
        model.add(Dense(units=55))
        model.add(Activation('relu'))
        model.add(Dense(units=45))
        model.add(Activation('relu'))
        model.add(Dense(units=35))
        model.add(Activation('relu'))
        model.add(Dense(units=self.distance-1))
        model.summary()
    
        model.compile(loss=self.losses[0], optimizer='adam', metrics=['mse','mae'])
        history = model.fit(self.X_train, self.y_train, epochs=5000, verbose=1, validation_split=0.2,callbacks=[self.callbacks[0]]) 
        model.evaluate(self.X_train, self.y_train)
        predict=model.predict(self.X_test)
    
        #charts(title,history,X_test,y_test,predict)
        self.metrics(predict,self.y_test)
        model.save("models/model_"+self.code+".dnn")
        
    def _get(self):
        K.clear_session()
        model = load_model('models/model_'+self.code+'.dnn')
        model.compile(loss=self.losses[0], optimizer=self.optimizers[5], metrics=['mse'])
        prediction = np.array([[self.weight]])
        pred = model.predict(prediction)
        pred = np.array(pred[0],dtype=float)
        pattern = self.Pattern(pred)
        print("{PRED:["+pattern+"]:PRED}")
        return pred
    
    def _get_center(self):
        K.clear_session()
        model = load_model('models/model_'+self.code+'.dnn')
        model.compile(loss=self.losses[0], optimizer=self.optimizers[5], metrics=['mse'])
        prediction = np.array([[1]])
        pred = model.predict(prediction)
        pred = np.array(pred[0],dtype=float)
        pattern = self.Pattern(pred)
        print("{PRES:["+pattern+"]:PRES}")
        return pred
    
    def _init(self):
        dataset = pd.read_csv(self.dataset, engine='python')
        y = dataset.iloc[:,1:self.distance].values
        X = dataset.iloc[:,[0]].values
        y = np.round(y,3)        
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=0.3)
      
    def metrics(self,y_pred,y_true):
        R2 = r2_score(y_true,y_pred)
        MSE = mean_squared_error(y_true,y_pred)
        RMSE = math.sqrt(MSE)
    
        print("------------------------------")
        print("R2=",R2)
        print("MSE=",MSE)        
        print("RMSE=",RMSE) 
        
    def charts(self):
        plt.style.use('seaborn')
        plt.scatter(_X1,_y, marker='.', c='b')
        plt.title('Model scatter')
        plt.ylabel('y')
        plt.xlabel('X')
        plt.legend(['train', 'validation'], loc='best')
        plt.savefig('./res/scatter_'+'scat'+'.png')
        plt.show()
    
        plt.style.use('seaborn')
        plt.scatter(_X2,_y, marker='.', c='b')
        plt.title('Model scatter')
        plt.ylabel('y')
        plt.xlabel('X2')
        plt.legend(['train', 'validation'], loc='best')
        plt.savefig('./res/scatter_'+'scat'+'.png')
        plt.show()  

    
    
class model_CAL(Methods):
    
    distance = None
    pressure = None
    meter = None
    
    def _set(self,distance,pressure,meter):
        self.distance = distance
        self.pressure = pressure
        self.meter = meter
        
    def _formule(self):
        x = np.array(self.distance, dtype=float)
        y = np.array(self.pressure, dtype=float)
        c,b = self.Power(x,y)
        formule = c*math.pow(self.meter,b)
        
        print("{c= "+str(c)+"}")
        print("{b= "+str(b)+"}")
        print("{y= "+str(formule)+" (m)}")

        
if __name__ == '__main__':
    
    import sys
    arg = sys.argv
    print("-Run...")
    
    print("-Arguments:"+str(arg))
        
    _action = arg[1]
    _operation = arg[2]
    _code = arg[3]
    _weight = arg[4]
    _meter = arg[5]
    _distance = int(arg[6])+1

    """_action = 1
    _operation = 1.2
    
    _code,_meter,_weight = '',0,0"""
    
    distance,pressure= None,None
    #distance = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5], dtype=float)
    
    model = model_DNN() 
    model._load("lab_"+_code,"dataset.csv",_weight,_distance)
    model._init()

    if(_action=='1'):
        if(_operation=='1.1'):
            print("-Learning...")
            model._set()
        elif(_operation=='1.2'):
            print("-Predicting...")
            pressure = model._get()
            pressure_dist = model._get_center()
                
                 
    """cal = model_CAL()
    cal._set(distance,pressure,meter)
    cal._formule()"""

    
    """plt.style.use('seaborn')
    plt.scatter(distance,pressure, marker='.', c='b')
    plt.title('Model scatter')
    plt.ylabel('pressure (kPa)')
    plt.xlabel('distance (m)')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()"""