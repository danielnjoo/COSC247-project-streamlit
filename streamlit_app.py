import pandas as pd
import numpy as np
import streamlit as st
from streamlit import components
import xgboost as xgb
import eli5
import pickle

"""
# COSC247 Project - Interactive
### Daniel Njoo

This is an app to accompany my Final Project, "(Machine) Learning to Detect Fake News," that allows users to explore the fully-trained XGBoost model's predictions, with regards to which words are most influential in making its predictions.

The project code can be found at https://github.com/danielnjoo/COSC247-project-streamlit.

A sample of articles, one from each source (n=66) is shown below:
"""

sample = pd.read_csv("sample.csv")
st.dataframe(sample[["source", "name", "content"]])
st.write("________________________________________________")
"""
## Selected article

Recall that in this case, label=1 refers to a trustworthy source while label=2 refers to an untrustworthy source (interpreted here as "fake news") as scored by NewsGuard.

Select an article in the sidebar on the left: and move the slider to show more of the article's text below:
"""

options = ("(" + sample["source"] + ") " + sample['name']).tolist()
content = sample['content'].tolist()
dic = dict(zip(content, options))

selected = st.sidebar.selectbox('Choose an article:', content, format_func=lambda x: dic[x])
st.write(sample.loc[sample["content"]==selected])

nchar = st.sidebar.slider("Number of characters to show:", min_value=0, max_value=len(selected), value=600)
st.write(selected[:nchar])
st.write("________________________________________________")

"""
## Model prediction
"""

Tfidf_vect_full = pickle.load(open("Tfidf_full.pickle.dat", "rb"))
xg = pickle.load(open("xgboost.pkl", "rb"))
xg.feature_names = Tfidf_vect_full.get_feature_names()

transformed_text = Tfidf_vect_full.transform([selected])
pred = xg.predict(transformed_text)
st.text("The model predicts " + np.array_str(pred) + " when the true label was " + sample.loc[sample["content"]==selected]["label"].to_string(index=False))
st.write("________________________________________________")

"""
## Prediction explanation

Using the eli5 package, the XGBoost model's prediction choice can be explained below by the following features, representing words in the Tf-Idf vectorizer (20,000 features only) that also appear in the text.

They are ranked by contribution that sum to the score. The bias refers to the intercept.
"""

no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
raw_html = eli5.show_prediction(xg, selected, vec=Tfidf_vect_full, show_feature_values=True, feature_filter=no_missing)._repr_html_()
components.v1.html(raw_html, height=500, scrolling=True)
