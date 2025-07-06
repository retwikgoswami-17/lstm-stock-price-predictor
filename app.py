import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Stock Price Predictor",layout="centered")
st.title("7-Day Stock Price Predictor")

base_dir=os.path.dirname(os.path.abspath(__file__))
dataset_dir=os.path.join(base_dir,"datasets")
model_dir=os.path.join(base_dir,"models")

datasets={"AAPL":os.path.join(dataset_dir,"AAPL.csv"),
          "ADBE":os.path.join(dataset_dir,"ADBE.csv"),
          "AMZN":os.path.join(dataset_dir,"AMZN.csv"),
          "BAC":os.path.join(dataset_dir,"BAC.csv"),
          "COST":os.path.join(dataset_dir,"COST.csv"),
          "CRM":os.path.join(dataset_dir,"CRM.csv"),
          "FB":os.path.join(dataset_dir,"FB.csv"),
          "GOOGL":os.path.join(dataset_dir,"GOOGL.csv"),
          "GS":os.path.join(dataset_dir,"GS.csv"),
          "IBM":os.path.join(dataset_dir,"IBM.csv"),
          "JNJ":os.path.join(dataset_dir,"JNJ.csv"),
          "JPM":os.path.join(dataset_dir,"JPM.csv"),
          "MA":os.path.join(dataset_dir,"MA.csv"),
          "MSFT":os.path.join(dataset_dir,"MSFT.csv"),
          "PEP":os.path.join(dataset_dir,"PEP.csv"),
          "PFE":os.path.join(dataset_dir,"PFE.csv"),
          "TGT":os.path.join(dataset_dir,"TGT.csv"),
          "V":os.path.join(dataset_dir,"V.csv"),
          "WMT":os.path.join(dataset_dir,"WMT.csv"),
          "XOM":os.path.join(dataset_dir,"XOM.csv"),
}

ticker=st.selectbox("Select Stock",list(datasets.keys()))
predict=st.button("Predict")

if predict:
    df=pd.read_csv(datasets[ticker])
    df.set_index("Date",inplace=True)
    df=df[['Open','High','Low','Close','Adj Close']].dropna()
    
    model_path=os.path.join(model_dir,ticker,"model.h5")
    scaler_path=os.path.join(model_dir,ticker,"scaler.pkl")
    
    model=load_model(model_path)
    data=joblib.load(scaler_path)
    scaler=data["scaler"]
    train_actual=data["train_actual"]
    train_pred=data["train_pred"]
    train_dates=pd.to_datetime(data["train_dates"])
    test_actual=data["test_actual"]
    test_pred=data["test_pred"]
    test_dates=pd.to_datetime(data["test_dates"])
    train_mape=mean_absolute_percentage_error(train_actual,train_pred)

    train_dates=pd.Series(train_dates)
    train_actual=pd.Series(train_actual)
    train_pred=pd.Series(train_pred)
    end_date=pd.to_datetime('2020-01-01')
    end=train_dates<=end_date

    st.subheader("Historical Trends of Stock Prices")
    fig1,ax1=plt.subplots()
    ax1.plot(train_dates[end],train_actual[end],label="Actual",color='blue',linewidth=0.5)
    ax1.plot(train_dates[end],train_pred[end],label="Predicted",color='orange',linewidth=0.5)
    ax1.set_title(f"{ticker}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    fig1.autofmt_xdate()
    st.pyplot(fig1) 
    st.markdown(f"Mean Absolute Percentage Error (MAPE): **{round(train_mape*100, 2)}%**")

    st.subheader("Actual vs Predicted Stock Prices")
    fig2,ax2=plt.subplots()
    ax2.plot(test_dates,test_actual,label="Actual",color='blue',linewidth=1)
    ax2.plot(test_dates,test_pred,label="Predicted",color='orange',linewidth=1)
    ax2.set_title(f"{ticker}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close Price")
    ax2.legend()
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    table=pd.DataFrame({
        "Date":test_dates.strftime('%d-%m-%y'),
        "Actual Price":np.round(test_actual,2),
        "Predicted Price":np.round(test_pred,2),
    }).set_index("Date")
    st.dataframe(table)

    test_mape=mean_absolute_percentage_error(test_actual,test_pred)
    st.markdown(f"Mean Absolute Percentage Error (MAPE): **{round(test_mape*100, 2)}%**")