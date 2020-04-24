import streamlit as st

x = st.slider("Level of x", 2, 20, 10, 1)
y = st.slider("Level of y", -20, 200, 10, 4)

st.write('x={}, y={}'.format(x, y))
