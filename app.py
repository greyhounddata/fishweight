import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Session state variables

if 'fish_stats' not in st.session_state:
    st.session_state.fish_stats = {
    'Length1':0, 
    'Length3':0,
    'Species_Bream':0, 
    'Species_Parkki':0, 
    'Species_Perch':0, 
    'Species_Pike':0,
    'Species_Roach':0, 
    'Species_Smelt':0, 
    'Species_Whitefish':0
    }

if "prediction" not in st.session_state:
    st.session_state.prediction= np.array([259.0])

# Functions
@st.cache
def load_data():
    all_data = pd.read_csv("data/Fish.csv")
    return all_data

def update_stats():
    
    st.session_state.fish_stats['Species_Bream']=0
    st.session_state.fish_stats['Species_Parkki']=0
    st.session_state.fish_stats['Species_Perch']=0
    st.session_state.fish_stats['Species_Pike']=0
    st.session_state.fish_stats['Species_Roach']=0
    st.session_state.fish_stats['Species_Smelt']=0
    st.session_state.fish_stats['Species_Whitefish']=0

    st.session_state.fish_stats['Length1'] =  st.session_state.length1
    st.session_state.fish_stats['Length3'] =  st.session_state.length3


    if st.session_state.species == 'Bream':
         st.session_state.fish_stats['Species_Bream']=1
    elif st.session_state.species =='Parkki':
         st.session_state.fish_stats['Species_Parkki']=1
    elif st.session_state.species =='Perch':
         st.session_state.fish_stats['Species_Perch']=1
    elif st.session_state.species =='Pike':
         st.session_state.fish_stats['Species_Pike']=1
    elif st.session_state.species =='Roach':
         st.session_state.fish_stats['Species_Roach']=1
    elif st.session_state.species =='Smelt':
         st.session_state.fish_stats['Species_Smelt']=1
    elif st.session_state.species =='Whitefish':
         st.session_state.fish_stats['Species_Whitefish']=1

    X = np.array(list(st.session_state.fish_stats.values())).reshape(1,-1)
    st.session_state.prediction = model.predict(X)
    
# Remove footer

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load Regression Model
filename = "models/gb_model.sav"
model = pickle.load(open(filename, 'rb'))

# Load data



# Page contents

st.title('Fish Weight Predictor Model')
st.subheader('Example of a deployed Machine Learning Regressor')
st.markdown("This model takes the vertical length and cross length for one of 7 species of fish, and predicts the fish weight.")
st.markdown("This is a project by [Greyhound Data](https://greyhounddata.github.io/)")
species = st.selectbox(
     'Species of fish:',
     ('Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'), on_change=update_stats, key="species" )

length1 = st.slider(
    "Vertical Length (cm)",
    min_value=5.0,
    max_value=80.0,
    value = 23.0,
    on_change = update_stats,
    key="length1"

)

length3 = st.slider(
    "Cross Length (cm)",
    min_value=5.0,
    max_value=80.0,
    value=30.0,
    on_change = update_stats,
    key="length3"
)


st.markdown("The predicted weight is: **{} grams**.".format(np.round(st.session_state.prediction[0])))

st.subheader("Plots")
st.markdown("See how your fish fits in with the training data...")


df = load_data()

fig1 = px.scatter(df, x='Length1', y='Weight', color='Species', labels={"Length1":"Vertical Length (cm)", "Length3": "Cross Length (cm)", "Weight":"Fish Weight (grams)"})
fig1.add_trace(go.Scatter(x=[st.session_state.length1], y=[st.session_state.prediction[0]], mode = 'markers',
                         marker_symbol = 'x', marker_color='black', name="User Input",
                         marker_size = 10))

fig2 = px.scatter(df, x='Length3', y='Weight', color='Species', labels={"Length1":"Vertical Length (cm)", "Length3": "Cross Length (cm)", "Weight":"Fish Weight (grams)"})
fig2.add_trace(go.Scatter(x=[st.session_state.length3], y=[st.session_state.prediction[0]], mode = 'markers',
                         marker_symbol = 'x', marker_color='black', name="User Input",
                         marker_size = 10))
st.write(fig1)
st.write(fig2)


st.subheader("Further Details")
st.markdown('''The model is a Gradient Boosted Regressor (scikit-learn implementation).  \n
On the holdout dataset we achieved: 
- *r2_score*: 0.946, 
- *Mean Absolute Error*: 39.8,
- *Mean Squared Error*: 7390
The main purpose of this project was to demonstrate a deployed machine learning model. 
The rest of the ML workflow can be seen in [this git repo](https://github.com/greyhounddata/fishweight), which includes a basic EDA and hyperparameter tuning. **However please keep in mind this is a small toy example and everything done here is an abridged version of what would be done in a real project.**

The training dataset can be found [here](https://www.kaggle.com/datasets/aungpyaeap/fish-market)
''')