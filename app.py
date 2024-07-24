import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('decision_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('CLASSIFICATION DATASET.csv')
# Extracting independent variable:
X = dataset.iloc[:, [0:14].values
# Encoding the Independent Variable
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value="Female", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,5:6]) 
#Replacing missing data with the calculated mean value  
X[:,5:6]= imputer.transform(X[:, 5:6])

imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value="Spain", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,6:7]) 
#Replacing missing data with the calculated mean value  
X[:,6:7]= imputer.transform(X[:, 6:7])

imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,[0,1,2,3,4,7,8,9,10,11,12,13]]) 
#Replacing missing data with the calculated mean value  
X[:,[0,1,2,3,4,7,8,9,10,11,12,13]]= imputer.transform(X[:, [0,1,2,3,4,7,8,9,10,11,12,13]]) 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
labelencoder_X = LabelEncoder()
X[:,6] = labelencoder_X.fit_transform(X[:,6])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict(age	,cp	,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal):
  output= model.predict(sc.transform([[age	,cp	,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
  print("Heart Disease", output)
  if output==[0]:
  print("It is heart disease of category 0")
  elif output==[1]:
  print("It is heart disease of category 1")
  elif output==[2]:
  print("It is heart disease of category 2")
  elif output==[3]:
  print("It is heart disease of category 3")
  elif output==[4]:
  print("It is heart disease of category 4")
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart disease Prediction using Decision Tree Classification")
    Age = st.text_input("age","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    cp  = st.number_input('Insert CP')
    trestbps = st.number_input('Insert trestbps')
    chol = st.number_input('Insert chol')
    fbs = st.number_input('Insert fbs')
    Geography =  st.number_input('Insert Geography')
    restecg=	 st.number_input('Insert restecg ')
    thalach =	 st.number_input('Insert thalach')
    exang=	 st.number_input('Insert exang')
    oldpeak=	 st.number_input('Insert oldpeak')
    slope =	 st.number_input('Insert slope')
    ca=	 st.number_input('Insert ca')
    thal= st.number_input('Insert thal')
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    
   
  
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(age	,cp	,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Deepak Moud")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()
