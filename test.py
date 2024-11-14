import pandas as pd

# Charger les données
df = pd.read_csv('Financial_inclusion_dataset.csv')
# Afficher les premières lignes
df
df.info()
df.describe()
#from ydata_profiling import ProfileReport

#profile = ProfileReport(df, title="Profiling Report")
#profile
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

df
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
df['uniqueid'] = label_encoder.fit_transform(df['uniqueid'])
df['location_type'] = label_encoder.fit_transform(df['location_type'])
df['gender_of_respondent'] = label_encoder.fit_transform(df['gender_of_respondent'])
df['relationship_with_head'] = label_encoder.fit_transform(df['relationship_with_head'])
df['marital_status'] = label_encoder.fit_transform(df['marital_status'])
df['education_level'] = label_encoder.fit_transform(df['education_level'])
df['bank_account'] = label_encoder.fit_transform(df['bank_account'])
df['cellphone_access'] = label_encoder.fit_transform(df['cellphone_access'])
df['job_type'] = label_encoder.fit_transform(df['job_type'])
df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x = df.drop('bank_account', axis=1)
y = df['bank_account']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

import streamlit as st

st.title("Prédiction d'inclusion financière")

feature1 = st.text_input("Entrez la Localisation du clien(Urban, Rural)")
feature2 = st.text_input("Entrez l'accés au téléphon(Yes, No)")

if st.button('Prédire'):
    prediction = model.predict([[feature1, feature2]])
    st.write(f"La prédiction est : {prediction}")
