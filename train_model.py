import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from keras.losses import MeanSquaredError

def train_and_save_model(ticker,csv_path,model_dir="models",seq_len=240,pred_len=7):
    
    df=pd.read_csv(csv_path,parse_dates=["Date"])
    df.set_index("Date",inplace=True)
    df=df[['Open','High','Low','Close','Adj Close']].dropna()

    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(df)

    train_data=scaled_data[:-pred_len]
    test_data=scaled_data[-pred_len:]
    test_dates=df.index[-pred_len:]

    X,Y=[],[]
    for i in range(seq_len,len(scaled_data)-pred_len+1):
        X.append(scaled_data[i-seq_len:i])
        Y.append(scaled_data[i:i+pred_len,3])

    X=np.array(X)
    Y=np.array(Y)

    model=Sequential([LSTM(50,activation='tanh',return_sequences=True,input_shape=(seq_len,df.shape[1])),
                      Dropout(0.2),
                      LSTM(50,activation='tanh'),
                      Dropout(0.2),
                      Dense(pred_len)
                      ])
    model.compile(optimizer='adam',loss=MeanSquaredError())
    model.fit(X,Y,epochs=50,batch_size=256,verbose=1)    
    
    train_actual,train_pred,train_dates=[],[],[]
    for i in range(seq_len,len(train_data)-pred_len+1):
        X_seq=train_data[i-seq_len:i]
        Y_true_seq=train_data[i:i+pred_len,3]
        Y_pred_seq_scaled=model.predict(X_seq.reshape(1,seq_len,df.shape[1]),verbose=0)[0]
        for j in range(pred_len):
            temp_pred=np.zeros(df.shape[1])
            temp_pred[3]=Y_pred_seq_scaled[j]
            inv_pred=scaler.inverse_transform([temp_pred])[0][3]
            train_pred.append(inv_pred)

            temp_true=np.zeros(df.shape[1])
            temp_true[3]=Y_true_seq[j]
            inv_true=scaler.inverse_transform([temp_true])[0][3]
            train_actual.append(inv_true)

            train_dates.append(df.index[i+j])

    train_mape=mean_absolute_percentage_error(train_actual,train_pred)
    print(f"{ticker} MAPE:{round(train_mape*100,2)}%")

    last_seq=scaled_data[-pred_len-seq_len:-pred_len]
    last_seq=last_seq.reshape(1,seq_len,df.shape[1])
    Y_test_scaled=model.predict(last_seq,verbose=0)[0]

    test_pred=[]
    for x in Y_test_scaled:
        temp=np.zeros(df.shape[1])
        temp[3]=x
        inv=scaler.inverse_transform([temp])[0][3]
        test_pred.append(inv)

    test_actual=[]
    for i in range(pred_len):
        temp=np.zeros(df.shape[1])
        temp[3]=test_data[i][3]
        inv=scaler.inverse_transform([temp])[0][3]
        test_actual.append(inv)

    test_mape=mean_absolute_percentage_error(test_actual,test_pred)
    print(f"{ticker} MAPE:{round(test_mape*100,2)}%")

    subdir=os.path.join(model_dir,ticker)
    os.makedirs(subdir,exist_ok=True)
    model.save(os.path.join(subdir,"model.h5"))
    joblib.dump({"scaler":scaler,
                 "train_actual":train_actual,
                 "train_pred":train_pred,
                 "test_actual":test_actual,
                 "test_pred":test_pred,
                 "train_dates":train_dates,             
                 "test_dates":list(test_dates)
    },os.path.join(subdir,"scaler.pkl"))

if __name__=="__main__":
    base_dir=os.path.dirname(os.path.abspath(__file__))
    dataset_path=os.path.join(base_dir,"datasets","AAPL.csv")
    model_path=os.path.join(base_dir,"models")
    train_and_save_model("AAPL",dataset_path,model_path)