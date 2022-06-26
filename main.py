#importing the dataset
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle

yes = [0,1]
no = [1,0]
Age_category = [ '18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69', '70-74'  ,'75-79' , '80 or older']
Race_list = ['American Indian/Alaskan Native' ,'Asian' ,'Black' ,'Hispanic' ,'Other','White'  ]
Diabetic_list = ['No' ,'No, borderline diabetes' ,'Yes', 'Yes (during pregnancy)']
Genhealth_list = ['Excellent','Fair' ,'Good' ,'Poor' ,'Very good']

st.markdown(
    """
    <style>
    .main{
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def read_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def load_file(filename,input):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)
    return loaded_model.predict([input])
def yes_no(x):
    if x == 'Yes' or x == 'Male':
        return yes
    else:
        return no
def findlist(x,xlist):
    list=[0]*len(xlist)
    list[xlist.index(x)]=1
    return list
def get_input():
    ip = smoking+alcohol+stroke+walking+gender+age+race+diabetic+physical_activity+general_health+asthma+kidney_disease+skin_cancer
    ip.append(bmi)
    ip.append(physical_illness)
    ip.append(mental_illness)
    ip.append(sleep)
    return ip


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Stroke Prediction using Logistic regression and Random Forest")


with dataset:
    #importing dataset
    st.header("Dataset that is being used in this project is Heart 2020 from kaggle")
    st.text("This is how dataset looks like")

    dataset = read_data('data/heart_2020_cleaned.csv')

    st.write(dataset.head(20))
    st.subheader("Data Characteristics")
    heart_problem = pd.DataFrame(dataset['HeartDisease'].value_counts())
    st.bar_chart(heart_problem)

    x = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values

    #encoding categorical data
    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2,3,6,7,8,9,10,11,12,14,15,16])],remainder="passthrough")
    x = np.array(ct.fit_transform(x))

    #encode dependent variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    #train test split
    X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)
 

with features:
    st.header("Features")
    st.markdown('* **First Feature**')
    st.markdown('* **Second Feature**')
    


with model_training:
    
    #logistic regression
    # classifier = LogisticRegression(random_state=0,solver='lbfgs', max_iter=1000)
    # classifier.fit(X_train,y_train)

    #random forest
    # classifier2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    # classifier2.fit(X_train, y_train)

    #prediction
    # y_pred = classifier.predict(X_test)
    # y_pred2 = classifier2.predict(X_test)

    #confusion matrix
    # cf = confusion_matrix(y_test,y_pred)
    # cf1 = confusion_matrix(y_test,y_pred2)
    # print(cf)
    # print(cf1)

    #accuracies
    # accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train,cv=10)
    # print(accuracies.mean())
    # print(accuracies.std())
    # accuracies2 = cross_val_score(estimator = classifier2, X=X_train, y=y_train,cv=10)
    # print(accuracies2.mean())
    # print(accuracies2.std())

    #st.text(classifier.predict([[0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,17,0,20,9]]))

    
    
    

    st.header("Model training using Logistic Regression and Random Forest")
    col1,col2 = st.columns(2)
    col1.subheader("Select Model")
    model = col1.selectbox('',['Logistic Regression', 'Random Forest'])

    col1.subheader("Select appropriate symptoms")
    col1.text('')

    bmi = float(col1.text_input("BMI value",value='0'))

    smoking = col1.selectbox("Smoking",['Yes','No'])
    smoking = yes_no(smoking)

    alcohol = col1.selectbox("AlcoholDrinking",['Yes','No'])
    alcohol = yes_no(alcohol)

    stroke = col1.selectbox("Stroke in Past",['Yes','No'])
    stroke = yes_no(stroke)

    physical_illness = int(col1.slider("Physical illness for how many days",max_value=30,min_value=0,value=0))
    
    mental_illness = int(col1.slider("Mental illness for how many days",max_value=30,min_value=0,value=0))

    walking = col1.selectbox("Difficulty in walking or climbing stairs",['Yes','No'])
    walking = yes_no(walking)

    gender = col1.selectbox("Gender",['Female','Male'])
    gender = yes_no(gender)

    age = col1.selectbox("Age Category",Age_category)#
    age = findlist(age,Age_category)
    print(age)
    
    race = col1.selectbox("Race",Race_list)#
    race = findlist(race,Race_list)
    print(race)
    
    diabetic = col1.selectbox("Diabetic",Diabetic_list)#
    diabetic = findlist(diabetic,Diabetic_list)
    print(diabetic)

    physical_activity = col1.selectbox("Physical Activity",['Yes','No'])
    physical_activity = yes_no(physical_activity)

    general_health = col1.selectbox("General Health",Genhealth_list)#
    general_health = findlist(general_health,Genhealth_list)
    print(general_health)

    sleep = int(col1.slider("Sleep Time ( 24 hr format )",max_value=24,min_value=1,value=20))

    asthma = col1.selectbox("Asthma",['Yes', 'No'])
    asthma = yes_no(asthma)

    kidney_disease = col1.selectbox("Kidney Disease",['Yes', 'No'])
    kidney_disease = yes_no(kidney_disease)

    skin_cancer = col1.selectbox("Skin cancer",['Yes', 'No'])
    skin_cancer = yes_no(skin_cancer)


    filename = 'finalized_model.sav'
    # pickle.dump(classifier, open(filename, 'wb'))
    input = get_input()
    print(input)
    output=load_file(filename,input)
    
    col2.subheader("Output of classifier:")
    
    if(output[0]==0):
        col2.subheader("No Heart Disease")
    else:
        col2.subheader("Yes there is chance for Heart Disease")
