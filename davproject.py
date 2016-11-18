# coding: utf-8
import numpy as np
import scipy.io as spio
data = spio.loadmat('data_CS306.mat')
data.keys()
a = data['data_144']
a = data['data_100']
b = data['data_144']
a_thresh = np.mean(a)
b_thresh = np.mean(b)
a_pdus = np.sum(a<a_thresh, axis=1)*100/24
b_pdus = np.sum(b<b_thresh, axis=1)*100/24
a_mos = np.mean(a, axis=1)
b_mos = np.mean(b, axis=1)
a_mos
b_mos
a_pdus
b_pdus
from sklearn.linear_model import LinearRegression, LogisticRegression
lingress = LinearRegression()
lingress.fit(a_mos.reshape(-1, 1), a_pdus)
lingress.score(a_mos.reshape(-1, 1), a_pdus)
np.mean(a)
np.mean(b)
get_ipython().magic('ls ')
a_mos
get_ipython().magic('ls ')
a_std = np.std(a, axis=1)
a_std.shape
a_std.shape
a_std
b_std = np.std(b, axis=1)
st.norm(a_mos, a_std)
import scipy.stats as st
st.norm(a_mos, a_std)
st.norm(a_mos, a_std).cdf(a_mos)
r = st.norm(a_mos, a_std)
r
st.norm(a_mos, a_std).cdf(a_mos+1)
st.norm(a_mos, a_std).cdf(3)
st.norm(a_mos, a_std).cdf(3).shape
st.norm(0, 1).cdf(3).shape
st.norm(0, 1).cdf(3)
st.norm(0, 1).cdf(0.3)
st.norm(a_mos, a_std).cdf(3).shape
st.norm(a_mos, a_std).cdf(3)
st.norm(a_mos, a_std).cdf(3)
st.norm(a_mos, a_std).cdf(3).shape
a.shape
a_mos.shape
a_mos[0]
a_std[0]
st.norm(a_mos[0], a_std[0]).cdf(3)
st.norm(a_mos[1], a_std[1]).cdf(3)
a_pdus[1]
np.var(a_pdus[1])
np.var(a_pdus)
a_pred = st.norm(a_mos, a_std).cdf(3)
np.var(a_pred)
np.var(a_pdus)
a_pdus
