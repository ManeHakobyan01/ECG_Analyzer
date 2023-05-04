import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
from ml_service import arrhythmia_classifier
import re
import heartpy as hp

import wfdb
import numpy as np
import torch
from PIL import Image


st.set_page_config(layout="wide")

def plot_graph(data, name = "raw", sampto = 3600):
    n_arr = np.array(data).reshape(-1,1)
    wfdb.wrsamp(name, fs = 360, units=['mV'], sig_name= ['MLII'], p_signal=n_arr, fmt=['80'], write_dir = f'ml_service/visualizations/')
    rec = wfdb.rdrecord(f'ml_service/visualizations/{name}', sampto = sampto)
    fig = wfdb.plot_wfdb(record=rec,  plot_sym=True,  time_units='seconds',  figsize=(20,8), ecg_grids='all',return_fig = True )
    fig.savefig(f'ml_service/visualizations/{name}.jpg')
    return fig  


def main():
    st.title('HiCare Cardio Hub')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    data = pd.read_csv(uploaded_file)  

    patient_number = st.selectbox('Patient id', (  [i for i in range(data.shape[0])] )) 

    lead = st.selectbox(
    'Select the lead',
    ('DI', 'DII', 'DIII' , 'AVL', 'AVF', 'AVR', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'MLII'))

    numbers = data[lead][patient_number]

    if numbers == 0:
        st.subheader('No record with the selected lead')
        return 0
    
    annotation = data.loc[patient_number]['label']
    numbers = re.sub(r'^\[|\]$', '', numbers) 
    full_data = [float(i) for i in numbers.split(', ')]

    integer_input = st.number_input(label='Scale:', format='%d', step=1, value=3600)
   

    df = pd.DataFrame({ 'measure': ['aa'], 'value': [0]})

    if st.button('Prediction and visualization'):
         if len(full_data) > 3600:
            name = "raw"
            plt = plot_graph(full_data, name, len(full_data))
            image = Image.open(f'ml_service/visualizations/{name}.jpg')
            st.image(image)

         col1, mid, col2 = st.columns([90,1,20])
         for i in range(len(full_data)//3600):
            data = full_data[i*3600: i*3600+3600] 
            data1 = torch.DoubleTensor([[ data  ]])  
            name = arrhythmia_classifier.predict(data1)
            names, probabilities = arrhythmia_classifier.predict_probabilities(data1)

            st.write("Prediction:    ", names[0], "with", "{:.2f}".format(probabilities[0]), "%")
            st.write("Prediction:    ", names[1], "with", "{:.2f}".format(probabilities[1]), '%')
            st.write("Prediction:    ", names[2], "with", "{:.2f}".format(probabilities[2]), '%')
            st.write("Annotation:    ", annotation)
            plt = plot_graph(data, name, integer_input )
            image = Image.open(f'ml_service/visualizations/{name}.jpg')
            with col1:
                st.image(image)
            wd, m = hp.process(data, 360)
            with col2:  
                i = 0
                for measure in m.keys():
                  df.loc[i] = measure, m[measure]
                  i+=1
                df = df.iloc[:2]
                st.table(df)
    

if __name__ == '__main__':
    main()