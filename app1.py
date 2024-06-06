import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

def explore():
    california_housing = fetch_california_housing()
    housing_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    housing_data['median_house_value'] = california_housing.target

    # Set the aesthetic style of the plots
    sns.set_style(style="whitegrid")

    # List of numeric columns to plot
    numeric_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
    'Latitude', 'Longitude', 'median_house_value']

    #map
    st.markdown("<h3 style='color: white;'><b>Map of Purchased Properties</b></h3>", unsafe_allow_html=True)
    fig = px.scatter_mapbox(housing_data, lat="Latitude", lon="Longitude", color="median_house_value",
                            color_continuous_scale=px.colors.sequential.Oranges, zoom=4, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)

    # Set up plotting area to display multiple histograms
    st.markdown("<h3 style='color: white;'><b>Histograms of Features</b></h3>", unsafe_allow_html=True)
    numeric_columns = housing_data.select_dtypes(include=np.number).columns.tolist()
    
    # Adjust the figure size and layout
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, column in enumerate(numeric_columns):
        sns.histplot(housing_data[column], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')
        
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    st.pyplot(fig)

     # Generate pairplots
    st.markdown("<h3 style='color: white;'><b>Pairplots of Features</b></h3>", unsafe_allow_html=True)
    fig = sns.pairplot(housing_data[numeric_columns], diag_kind='kde', height=2)
    st.pyplot(fig)

    # Scatter plot
    st.markdown("<h3 style='color: white;'><b>Median House Value vs Median Income</b></h3>", unsafe_allow_html=True)
    fig = px.scatter(housing_data, x="MedInc", y="median_house_value",color='median_house_value', color_continuous_scale=px.colors.sequential.Oranges)
    st.plotly_chart(fig)


def main():
    # Add custom CSS to change the background color of the entire page and sidebar
    st.markdown(
        """
        <style>
        body {
            background-color: black;
        }
        .stApp {
            background-color: black;
        }
        .sidebar .sidebar-content {
            background-color: black;
        }
        .stButton>button {
            color: navy;
        }
        .stTextInput>div>div>input {
            color: black;
        }
        h1, .st-sidebar h1 {
            color: navy;
        }
        div[data-testid="stHorizontalBlock"] label {
        color: white;
        font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with title
    st.sidebar.title("Menu")
    show_inputs = st.sidebar.checkbox("Real Estate Value Checker")
    dash = st.sidebar.checkbox("Dashboards")

    if not show_inputs and not dash:
        # Updated CSS for navy blue background header
        style = """<div style='background-color:navy; padding:12px'>
                   <h1 style='color:white'>The California Dream Real Estate</h1>
                </div>"""
        st.markdown(style, unsafe_allow_html=True)

        # Specify the path to your local image file
        image_path = "image_altered1.jpg"
        st.image(image_path, use_column_width=True)
    elif dash:
        # Updated CSS for navy blue background header
        style = """<div style='background-color:navy; padding:12px'>
                   <h1 style='color:white'>The California Dream Real Estate</h1>
                </div>"""
        st.markdown(style, unsafe_allow_html=True)
        explore()
    else:
        # Updated CSS for navy blue background header
        style = """<div style='background-color:navy; padding:12px'>
                   <h1 style='color:white'>The California Dream Real Estate</h1>
                </div>"""
        st.markdown(style, unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'><b>Real Estate Value Checker</b></h3>", unsafe_allow_html=True)
        left, right = st.columns((2, 2))
        med_inc = left.number_input('Median income (in tens of thousands of dollars)', step=0.1, format="%.2f", value=3.00)
        house_age = right.number_input('Median age of the building', step=1.0, format='%.1f', value=25.0)
        ave_rooms = left.number_input('Average number of rooms', step=1.0, format='%.1f', value=6.0)
        ave_bedrms = right.number_input('Average number of bedrooms', step=1.0, format='%.1f', value=1.0)
        population = left.number_input('Population', step=1.0, format='%.1f', value=100.0)
        ave_occup = right.number_input('Average number of occupants', step=1.0, format='%.1f', value=3.0)
        latitude = left.number_input('Latitude', step=0.01, format="%.2f", value=37.88)
        longitude = right.number_input('Longitude', step=0.01, format="%.2f", value=-122.23)

        button = st.button('Predict')

        # if button is pressed
        if button:
            # make prediction
            result = predict(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude)
            st.success(f'The estimated value of the house is ${round(result, 2)}')

# load the trained model
with open('rf_model.pkl', 'rb') as rf:
    model = pickle.load(rf)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)

def predict(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    # processing user input
    features = [med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]
    df = pd.DataFrame([features], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    # scaling the data
    df_scaled = scaler.transform(df)
    # making predictions using the trained model
    prediction = model.predict(df_scaled)
    result = prediction[0] * 100000  # Converting to actual house value in dollars
    return result

if __name__ == '__main__':
    main()
    
    with st.sidebar:
        # ... (existing sidebar content)
        
        st.markdown("---")  # Add a horizontal line separator
        
        # Add the custom text with Streamlit logo and CDRE
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by CDRE (The California Dream Real Estate Co)</h6>',
            unsafe_allow_html=True,
        )