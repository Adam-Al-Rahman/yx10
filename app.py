import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.express as px
from PIL import Image
import pymongo

img=Image.open("./images/favicon.png")
st.set_page_config(page_title="y = x + 10", page_icon=img)

# Initialize connection.
client = pymongo.MongoClient(st.secrets.MONGO_CLIENT)

# Database
tea_milk_coffee = client["tea_milk_coffee"]


sidebar = st.container()
header = st.container()
dataset = st.container()
model_training = st.container()
download_model=st.container()
footer = st.container()



@st.cache
def get_data(filename):
    data=pd.read_csv(filename)
    return data

def graph(graph):
    line_graph  = st.line_chart(graph)
    return line_graph

with header:
    st.write(
    """
    # Linear Regression ML Model

    In this we'll see that how a machine learning model will predict the value of `y` and satisfy the line equation `y = mx + c`.
    """)


with dataset:
    st.header("Number Array Dataset")
    st.write(
    """
        The dataset is the simple float numbers. I created this data. So, if you wanna download the data click the below button `DataSet`.
    """)

    data=get_data('./data/y=x+10.csv')

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(data)

    st.download_button(
        label="DataSet",
        data=csv,
        file_name='y=x+10.csv',
        mime='text/csv',
    )

    st.write("""
        >  `Note`: While training the model, data was split into three set [train, validation and test]
    """)

    data_xy=pd.DataFrame(data).drop(['ID'], axis = 1)
    st.write(data.head(30))
    st.write(" ")
    st.subheader("Visualizing the Data")
    graph(data_xy)


with model_training:
    st.header("Model Prediction")

    # Loading the  model ymc.h5

    model = tf.keras.models.load_model("./model/ymc.h5")


    sel_col, disp_col = st.columns(2)
    sel_col.write(" ")
    sel_col.write(" ")
    sel_col.write(" ")
    x_value = sel_col.slider('Select for [ x ]', min_value=-10000.0, max_value=10000.0, value=2100.0)
    y_pred = tf.squeeze(model.predict([x_value]))

    disp_col.subheader("Model Predicted Value" )
    disp_col.code(f"y = {y_pred}")

    y_true=x_value+10

    disp_col.subheader("Actual Value")
    disp_col.code(f"y = {y_true}")

    disp_col.subheader("Mean Absolute Error")
    disp_col.code(f"MAE = {tf.squeeze(tf.keras.losses.mean_absolute_error(y_true, model.predict([x_value])))}")

    fig = px.scatter()


    fig.add_scatter(
        x=[x_value],
        y=[y_pred],
        name="Predicted Value"

    )

    fig.add_scatter(
        x=[x_value],
        y=[y_true],
        name="Actual Value",
    )

    st.subheader("Predicted vs. Actual")
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """

        ### Model Summary
        ##### Model: "yx10"

        |Layer (type)               |  Output Shape          |    Param #|
        |---|---|---|
        |input_layer_1 (Dense)   |     (None, 60)         |       120 |
        |input_layer_2 (Dense)    |    (None, 50)           |     3050 |
        |Output_Layer (Dense)    |     (None, 1)            |     51 |
        | | | |
        |Total params: 3,221|
        |Trainable params: 3,221|
        |Non-trainable params: 0|
        """
    )

    st.write(" ")
    st.subheader("Model Neural Network")
    st.image("./images/ymc.jpg")
    # st.write("""
    #     ##### View model's neural network in detail click `Neural Network`.
    # """)

    st.write(" ")

with sidebar:
    st.sidebar.markdown("""
    <style>
    details > summary {
    list-style: none;
    }
    details > summary::-webkit-details-marker {
    display: none;
    }
    </style>
    <details>
        <summary style="color:#bf616a;-webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; -o-user-select: none; user-select: none; margin-bottom:15px;cursor:pointer">
            <h1 align="center"><span  style="color:pink; border: 3px; border-color: #00aa00; border-style: dashed;padding:5px;border-radius:4px;">y = x + 10</span></h1>
        </summary>
            <h4 align="center">Line Equation: y = mx + c</h4>
    </details>

    ---
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <h3 align="center">
    Let, <code>m = 1</code> | <code>c = 10`<br/></code>
    <code>selected [x]: {x_value}</code></br>
    Equation: <code>y = {x_value} + 10</code>
    </h3>
    """,unsafe_allow_html=True)

    st.sidebar.image("./images/logo.png")


with download_model:

    st.subheader("Download Model")

    with st.form("tea_cappuccino_coffee", clear_on_submit=True):
        st.code("Submit the below form to download the ml model.")
        nick_name=st.text_input('Nick Name', placeholder="Johnny")
        poision=st.selectbox('What you like', ['Tea', 'Milk', 'Coffee', 'Choose a option'], help="What you like in drinks", index=3)

        submitted = st.form_submit_button("Submit")

        if poision=="Choose a option":
            pass

        if nick_name and poision and not poision=="Choose a option":
            if submitted:
                st.write("""
                    <h6 align="center">Thank you!</h6>
                <h6 align="center">Download by clicking below button.</h6>
                """, unsafe_allow_html=True)

        # collections
        collections = tea_milk_coffee["users"]

        if poision!="Choose a option":
            user_info = { "name": f"{nick_name}" , "drink": f"{poision}"}
            x = collections.insert_one(user_info)

    if nick_name and poision and not poision=="Choose a option":
        if submitted:
            with open("./model/ymc.h5", "rb") as file:
                btn = st.download_button(
                    label="Download Model",
                    data=file,
                    file_name="yx10.h5"
                )



    load_col, down_col  = st.columns(2)

    down_col.write(
        """
        ---

        Check out the Source Code:

        ###### <span  style="color:pink; border: 2px; border-color: #261799; border-style: dashed;padding:3px;border-radius:4px;"><a href="https://adam-al-rahman.github.io/LabGarden/neural-network/regression/machine_learning/tensorflow/ml_model/2022/01/02/straight-line.html" target="_blank" style=" text-decoration: none; color:pink">Source Code</a></span>

        By <a href="https://atiq-ur-rehaman.netlify.app/" target="_blank" style="text-decoration: none; color:#00aa00">Adam Al-Rahman</a>

        """, unsafe_allow_html=True
    )


with footer:

    # To hide "made with streamlit"
    hide_streamlit_style ="""
    <style>

        #MainMenu {visibility: hidden;} <!--- To hide the Hamburger icon --->
        footer {visibility: hidden;}

        footer {
            visibility: hidden;
        }

        footer:after {
            content:'Build by Adam Al-Rahman';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }

    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
