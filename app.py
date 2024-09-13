import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


def load_data():
    data = pd.read_csv('data\BDM.csv')
    return data

def train_model(data):
    X = data.drop('Sales', axis=1)
    y = data['Sales']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    model.fit(X, y)
    joblib.dump(model, 'sales_predictor.pkl')
    return model


def load_model():
    return joblib.load('preprocessor\sales_predictor.pkl')

data = load_data()
model = load_model()

def main():
    st.title("Sales Prediction")

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    categories = ['Crockery', 'Gents', 'Ladies', 'Toys']
    
    crockery_products = ['Bowls', 'Cups', 'Dinner_Set', 'Plates']
    toy_products = [ 'Soft_Toys','Teddy_Bears', 'Electronic_Toys', 'Electric_Cars']
    ladies_products = ['Bags', 'Clothings','Footwear','Accessories']
    gents_products = ['Shirt', 'Trousers', 'Jackets', 'Shoes','Sunglasses', 'Belts', 'Wallets']
    
    month = st.selectbox('Select Month', months)
    year = st.number_input("Year")
    
    category = st.selectbox('Select Category', categories)
    
    if category == 'Crockery': 
        product = st.selectbox('Select Product', crockery_products)
    elif category == 'Gents': 
        product = st.selectbox('Select Product', gents_products)
    elif category == 'Ladies': 
        product = st.selectbox('Select Product', ladies_products)
    elif category == 'Toys': 
        product = st.selectbox('Select Product', toy_products)

    price = st.number_input("Price")

    if st.button('Predict'):
        columns = ['Month', 'Year', 'Category', 'Product', 'Price']
        data = [[month, year, category, product, price]]
        data_df = pd.DataFrame(data=data,columns=columns)
        st.write(data_df)
    
        if model is not None:
            with st.spinner('Predicting...'):
                prediction = model.predict(data_df)
                prediction = prediction.astype(int)
                    
            st.markdown(f'<h1 style="font-size:48px; color:green">Predicted Sales of Product is: {prediction}</h1>', unsafe_allow_html=True)
if __name__ == '__main__':
    main()