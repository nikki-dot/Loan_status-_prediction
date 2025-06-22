import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Loan Data', layout="wide")

st.markdown('''<style>
div.block-container{padding:2.5rem}
</style>''', unsafe_allow_html=True)

st.title(":moneybag: Loan Status Prediction")

classifier = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('labelencoder.pkl', 'rb'))


gender = st.selectbox(":man_and_woman_holding_hands: Gender",['Male', 'Female'])
gender_encoded = encoder['Gender'].transform([gender])[0]
married = st.selectbox(":couple_with_heart: Married",['Yes','No'])
married_encoded = encoder['Married'].transform([married])[0]
dependents = st.number_input(':man-boy-boy: Dependents', min_value=0)
education = st.selectbox(":books: Education",['Graduate', 'Not Graduate'])
edu_encoded = encoder['Education'].transform([education])[0]
employed = st.selectbox(":factory_worker: Self_Employed",['Yes','No'])
employed_encoded = encoder['Self_Employed'].transform([employed])[0]
applicant = st.number_input(':dollar: ApplicantIncome', min_value=1000)
coapplicant = st.number_input('CoapplicantIncome', min_value=0.0)
amount = st.number_input('LoanAmount', min_value=0.0)
term = st.number_input('Loan_Amount_Term', min_value=30.0)
history = st.number_input('Credit_History', min_value=0)
area = st.selectbox("Property_Area",['Urban', 'Rural', 'Semiurban'])
area_encoded = encoder["Property_Area"].transform([area])[0]

input_data = (gender_encoded, married_encoded, dependents, edu_encoded, employed_encoded, applicant, coapplicant, amount, term, history, area_encoded)
finaldata = np.asarray(input_data, dtype=np.float32).reshape(1, -1)
output = classifier.predict(finaldata)

if st.button("Predict"):

    if output[0] == 0:
        st.header("Loan Rejected")
    else:
        st.header("Loan Approved")









