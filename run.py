import scipy.io as spio
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.optimize import minimize

data = spio.loadmat('data_CS306.mat')

a = data['data_144']
b = data['data_100']

# Taking mean as the thresholds
a_thresh = 3#np.mean(a)
b_thresh = 3#np.mean(b)

a_pdus = np.sum(a<a_thresh, axis=1)*100/a.shape[1]
b_pdus = np.sum(b<b_thresh, axis=1)*100/b.shape[1]

a_mos = np.mean(a, axis=1)
b_mos = np.mean(b, axis=1)

a_std = np.std(a, axis=1)
b_std = np.std(b, axis=1)

plt.figure()
p, q = np.polyfit(a_mos, a_pdus, deg=1)
plt.plot(a_mos, p*a_mos + q, color='green')
plt.scatter(a_mos, a_pdus)  #, color='r', alpha=0.9)
plt.title("HDR Videos")#Mean Opinion Scores against Percentage of Dissatisfied Users")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")

plt.figure()
p, q = np.polyfit(a_mos, a_pdus, deg=1)
plt.plot(b_mos, p*b_mos + q, color='green')
plt.scatter(b_mos, b_pdus)  #, color='b', alpha=0.2)
plt.title("Full HD Videos")#Mean Opinion Scores against Percentage of Dissatisfied Users")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")

a_corr = st.pearsonr(a_mos, a_pdus)
b_corr = st.pearsonr(b_mos, b_pdus)

print("\n")
print("-"*21, "Linear Correlation Coefficients between MOS and PDUs", "-"*22)
print("HDR Videos:", a_corr[0])
print("Full HD Videos:", b_corr[0])

def f_score(x, y):
    v_res_x = np.var(x)
    v_res_y = np.var(y)
    df_x = len(x) - 1
    df_y = len(y) - 1
    if v_res_x > v_res_y:
        f = v_res_x/v_res_y
    else:
        f = v_res_y/v_res_x
    return f, st.f.cdf(f, df_x, df_y)

print("\n")
print('-'*26, "Model Testing Results | (F-Score, p-value)", '-'*27)

#### Linear Regression ####
lingress = LinearRegression()

lingress.fit(a_mos.reshape(-1, 1), a_pdus)#a_pdus.reshape(-1, 1))
a_lingress_pred_pdus = lingress.predict(a_mos.reshape(-1, 1))
a_f = f_score(a_lingress_pred_pdus, a_pdus)

lingress.fit(b_mos.reshape(-1, 1), b_pdus)#b_pdus.reshape(-1, 1))
b_lingress_pred_pdus = lingress.predict(b_mos.reshape(-1, 1))
b_f = f_score(b_lingress_pred_pdus, b_pdus)

print("-"*36, "Linear Regression Model", "-"*36)
print("HDR Videos:", a_f)
print("Full HD Videos:", b_f)

#### Logistic Regression ####

def logress(x, alphas):
    return (alphas[0]*(0.5 - 1/(1+np.exp(alphas[1]*(x-alphas[2])))) + alphas[3])

def a_loss_function(alphas):
    return np.sum( (a_pdus - logress(a_mos, alphas))**2 )

def b_loss_function(alphas):
    return np.sum( (b_pdus - logress(b_mos, alphas))**2 )

a_alphas = minimize(a_loss_function, [10, 1, 3, 1]).x
a_logress_pred_pdus = logress(a_mos, a_alphas)

b_alphas = minimize(b_loss_function, [10, 1, 3, 10]).x
b_logress_pred_pdus = logress(b_mos, b_alphas)

#logress = LogisticRegression()
#logress.fit(a_mos.reshape(-1, 1), a_pdus.ravel())#a_pdus.reshape(-1, 1))
#
#a_pred_pdus = logress.predict(a_mos.reshape(-1, 1))
#
#logress.fit(b_mos.reshape(-1, 1), b_pdus.ravel())#b_pdus.reshape(-1, 1))
#b_pred_pdus = logress.predict(b_mos.reshape(-1, 1))

a_f = f_score(a_logress_pred_pdus, a_pdus)
b_f = f_score(b_logress_pred_pdus, b_pdus)

print("-"*35, "Logistic Regression Model", "-"*35)
print("HDR Videos:", a_f)
print("Full HD Videos:", b_f)

#### Gaussian Model ####
#a_gauss_pred_pdus = 100*st.norm(a_mos, a_std).cdf(a_thresh)
#b_gauss_pred_pdus = 100*st.norm(b_mos, b_std).cdf(b_thresh)

gpr = GaussianProcessRegressor()

gpr.fit(a_mos.reshape(-1, 1), a_pdus)
a_gauss_pred_pdus = gpr.predict(a_mos.reshape(-1, 1))

gpr.fit(b_mos.reshape(-1, 1), b_pdus)
b_gauss_pred_pdus = gpr.predict(b_mos.reshape(-1, 1))

a_f = f_score(a_gauss_pred_pdus, a_pdus)
b_f = f_score(b_gauss_pred_pdus, b_pdus)

print("-"*40, "Gaussian Model", "-"*41)
print("HDR Videos:", a_f)
print("Full HD Videos:", b_f)

def mse(x, y):
    return np.mean( (x-y)**2 )

#### Training on one, testing on other ####

print("\n")
print("-"*15, "Training on HDR and Predicting Full HD PDUs | Mean Squared Errors", "-"*15)

#### Linear Model ####
a_lingress_pred_pdus = lingress.predict(a_mos.reshape(-1, 1))   #The model is already fit with b
a_lingress_mse = mse(a_lingress_pred_pdus, a_pdus)
print("Linear Model:", a_lingress_mse)

#### Logistic Model ####
a_logress_pred_pdus = logress(a_mos, b_alphas)  #b_alphas have already been calculated
a_logress_mse = mse(a_logress_pred_pdus, a_pdus)
print("Logistic Model:", a_logress_mse)

#### Gaussian Model ####
#a_gauss_pred_pdus = 100*st.norm(a_mos, a_std).cdf(a_thresh) #NEED TO CHECK!!!
a_gauss_pred_pdus = gpr.predict(a_mos.reshape(-1, 1))   #The model is already fit with b
a_gauss_mse = mse(a_gauss_pred_pdus, a_pdus)
print("Gaussian Model:", a_gauss_mse)
print()

#plt.show()
