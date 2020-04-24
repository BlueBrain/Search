import numpy as np
import time
import streamlit as st


@st.cache(allow_output_mutation=True)
def get_previous_state():
    return []


# END OF EXPERIMENTAL
@st.cache(show_spinner=False)
def backend(model, k, show_paragraph, query, require_journal, date_range, deprioritize, strength):
    time.sleep(1)

    return ['result_{}'.format(i + 1) for i in range(np.random.randint(2, 20))]


header = st.header('Blue Brain Search')
model = st.radio('Choose your model', options=['A', 'B'])
k = st.slider("Top k results", 1, 20, 10, 1)
show_paragraph = st.checkbox('Show paragraph')
query = st.text_area('Query', 'Glucose is a risk factor for COVID-19')
require_journal = st.checkbox('Require journal')
date_range = st.slider('Date range', 2010, 2020, (2015, 2020))
deprioritize = st.text_area('Deprioritize', '')
strength = st.radio('Deprioritization strength', options=['None', 'Mild', 'Stronger'])

investigate = st.button('Investigate')

all_variables = {'model': model,
                 'k': k,
                 'show_paragraph': show_paragraph,
                 'query': query,
                 'require_journal': require_journal,
                 'date_range': date_range,
                 'deprioritize': deprioritize,
                 'strength': strength}

if investigate:
    with st.spinner('Searching!'):
        result = backend(**all_variables)
        state = get_previous_state()
        state.clear()
        state.extend(result)

for row in get_previous_state():
    st.write(row)
