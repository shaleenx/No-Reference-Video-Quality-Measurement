import scipy.io as spio
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.optimize import minimize

import scikits.bootstrap as bootstrap

data = spio.loadmat('data_CS306.mat')

a = data['data_144']
b = data['data_100']

# Taking 3 as the threshold
a_thresh = 3
b_thresh = 3

a_pdus = np.sum(a<a_thresh, axis=1)*100/a.shape[1]
b_pdus = np.sum(b<b_thresh, axis=1)*100/b.shape[1]

a_mos = np.mean(a, axis=1)
b_mos = np.mean(b, axis=1)

a_std = np.std(a, axis=1)
b_std = np.std(b, axis=1)

a_corr = st.pearsonr(a_mos, a_pdus)
b_corr = st.pearsonr(b_mos, b_pdus)

plt.figure()
p, q = np.polyfit(a_mos, a_pdus, deg=1)
plt.plot(a_mos, p*a_mos + q, color='green')
plt.scatter(a_mos, a_pdus)
plt.title("HDR Videos")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")
plt.annotate("Pearson Correlation Coefficient: "+str(a_corr[0]), xy=(3.5, 80))

plt.figure()
p, q = np.polyfit(a_mos, a_pdus, deg=1)
plt.plot(b_mos, p*b_mos + q, color='green')
plt.scatter(b_mos, b_pdus)
plt.title("Full HD Videos")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")
plt.annotate("Pearson Correlation Coefficient: "+str(b_corr[0]), xy=(4, 80))

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

lingress.fit(a_mos.reshape(-1, 1), a_pdus)
a_lingress_pred_pdus = lingress.predict(a_mos.reshape(-1, 1))
a_f = f_score(a_lingress_pred_pdus, a_pdus)

lingress.fit(b_mos.reshape(-1, 1), b_pdus)
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

a_alphas = minimize(a_loss_function, [1000, 2, 2, 1]).x
a_logress_pred_pdus = logress(a_mos, a_alphas)

b_alphas = minimize(b_loss_function, [10, 1, 3, 10]).x
b_logress_pred_pdus = logress(b_mos, b_alphas)

a_f = f_score(a_logress_pred_pdus, a_pdus)
b_f = f_score(b_logress_pred_pdus, b_pdus)

print("-"*35, "Logistic Regression Model", "-"*35)
print("HDR Videos:", a_f)
print("Full HD Videos:", b_f)

#### Gaussian Model ####

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

a_lingress_mse = mse(a_lingress_pred_pdus, a_pdus)
a_logress_mse = mse(a_logress_pred_pdus, a_pdus)
a_gauss_mse = mse(a_gauss_pred_pdus, a_pdus)
b_lingress_mse = mse(b_lingress_pred_pdus, b_pdus)
b_logress_mse = mse(b_logress_pred_pdus, b_pdus)
b_gauss_mse = mse(b_gauss_pred_pdus, b_pdus)

print()
print("-"*40, "Self-fit MSE", "-"*40)
print()
print("LINEAR MODEL")
print("HDR:", a_lingress_mse)
print("Full HD:", b_lingress_mse)
print()
print("LOGISTIC MODEL")
print("HDR:", a_logress_mse)
print("Full HD:", b_logress_mse)
print()
print("GAUSSIAN MODEL")
print("HDR:", a_gauss_mse)
print("Full HD:", b_gauss_mse)

plt.figure()
plt.scatter(a_mos, a_pdus, color='red', label='Actual Values')
plt.scatter(a_mos, a_lingress_pred_pdus, color='black', label='Linear Regression | MSE: '+str(a_lingress_mse), alpha=0.8)
plt.scatter(a_mos, a_logress_pred_pdus, color='blue', label='Logisitic Regression | MSE: '+str(a_logress_mse), alpha=0.8)
plt.scatter(a_mos, a_gauss_pred_pdus, color='green', label='Gaussian Process Regression | MSE: '+str(a_gauss_mse), alpha=0.8)
plt.title("HDR Videos")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")
plt.legend()

plt.figure()
plt.scatter(b_mos, b_pdus, color='red', label='Actual Values')
plt.scatter(b_mos, b_lingress_pred_pdus, color='black', label='Linear Regression | MSE: '+str(b_lingress_mse), alpha=0.8)
plt.scatter(b_mos, b_logress_pred_pdus, color='blue', label='Logisitic Regression | MSE: '+str(b_logress_mse), alpha=0.8)
plt.scatter(b_mos, b_gauss_pred_pdus, color='green', label='Gaussian Process Regression | MSE: '+str(b_gauss_mse), alpha=0.8)
plt.title("Full HD Videos")
plt.xlabel("Mean Opinion Score")
plt.ylabel("Percentage of Dissatisfied Users")
plt.legend()

#### Training on Full HD, testing on HDR ####

print("\n")
print("-"*15, "Training on HDR and Predicting Full HD PDUs | Mean Squared Errors", "-"*15)

#### Linear Model ####
anew_lingress_pred_pdus = lingress.predict(a_mos.reshape(-1, 1))   #The model is already fit with b
anew_lingress_mse = mse(anew_lingress_pred_pdus, a_pdus)
print("Linear Model:", anew_lingress_mse)

#### Logistic Model ####
anew_logress_pred_pdus = logress(a_mos, b_alphas)  #b_alphas have already been calculated
anew_logress_mse = mse(anew_logress_pred_pdus, a_pdus)
print("Logistic Model:", anew_logress_mse)

#### Gaussian Model ####
#anew_gauss_pred_pdus = 100*st.norm(anew_mos, anew_std).cdf(anew_thresh) #NEED TO CHECK!!!
anew_gauss_pred_pdus = gpr.predict(a_mos.reshape(-1, 1))   #The model is already fit with b
anew_gauss_mse = mse(anew_gauss_pred_pdus, a_pdus)
print("Gaussian Model:", anew_gauss_mse)
print()

a_mean_linear_low, a_mean_linear_high = bootstrap.ci(data=(a_lingress_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)
a_var_linear_low, a_var_linear_high = bootstrap.ci(data=(a_lingress_pred_pdus-a_pdus)**2, statfunction=np.var, alpha=0.05)

b_mean_linear_low, b_mean_linear_high = bootstrap.ci(data=(b_lingress_pred_pdus-b_pdus)**2, statfunction=np.mean, alpha=0.05)
b_var_linear_low, b_var_linear_high = bootstrap.ci(data=(b_lingress_pred_pdus-b_pdus)**2, statfunction=np.var, alpha=0.05)

a_mean_logistic_low, a_mean_logistic_high = bootstrap.ci(data=(a_logress_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)
a_var_logistic_low, a_var_logistic_high = bootstrap.ci(data=(a_logress_pred_pdus-a_pdus)**2, statfunction=np.var, alpha=0.05)

b_mean_logistic_low, b_mean_logistic_high = bootstrap.ci(data=(b_logress_pred_pdus-b_pdus)**2, statfunction=np.mean, alpha=0.05)
b_var_logistic_low, b_var_logistic_high = bootstrap.ci(data=(b_logress_pred_pdus-b_pdus)**2, statfunction=np.var, alpha=0.05)

a_mean_gauss_low, a_mean_gauss_high = bootstrap.ci(data=(a_gauss_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)
a_var_gauss_low, a_var_gauss_high = bootstrap.ci(data=(a_gauss_pred_pdus-a_pdus)**2, statfunction=np.var, alpha=0.05)

b_mean_gauss_low, b_mean_gauss_high = bootstrap.ci(data=(b_gauss_pred_pdus-b_pdus)**2, statfunction=np.mean, alpha=0.05)
b_var_gauss_low, b_var_gauss_high = bootstrap.ci(data=(b_gauss_pred_pdus-b_pdus)**2, statfunction=np.var, alpha=0.05)

print("-"*25, "Bootstrap | (CI_low, MSE, CI_high)", "-"*25)
print("-"*40, "HDR", "-"*40)
print("LINEAR MODEL")
print("MSE:", a_mean_linear_low, mse(a_lingress_pred_pdus, a_pdus), a_mean_linear_high)
#print("VARIANCE:", a_var_linear_low, np.var(a_lingress_pred_pdus), a_var_linear_high)

print("LOGISTIC MODEL")
print("MSE:", a_mean_logistic_low, mse(a_logress_pred_pdus, a_pdus), a_mean_logistic_high)
#print("VARIANCE:", a_var_logistic_low, np.var(a_logress_pred_pdus), a_var_logistic_high)

print("GAUSSIAN MODEL")
print("MSE:", a_mean_gauss_low, mse(a_gauss_pred_pdus, a_pdus), a_mean_gauss_high)
#print("VARIANCE:", a_var_gauss_low, np.var(a_gauss_pred_pdus), a_var_gauss_high)

print("-"*40, "FULL HD", "-"*40)
print("LINEAR MODEL")
print("MSE:", b_mean_linear_low, mse(b_lingress_pred_pdus, b_pdus), b_mean_linear_high)
#print("VARIANCE:", b_var_linear_low, np.var(b_lingress_pred_pdus), b_var_linear_high)

print("LOGISTIC MODEL")
print("MSE:", b_mean_logistic_low, mse(b_logress_pred_pdus, b_pdus), b_mean_logistic_high)
#print("VARIANCE:", b_var_logistic_low, np.var(b_logress_pred_pdus), b_var_logistic_high)

print("GAUSSIAN MODEL")
print("MSE:", b_mean_gauss_low, mse(b_gauss_pred_pdus, b_pdus), b_mean_gauss_high)
#print("VARIANCE:", b_var_gauss_low, np.var(b_gauss_pred_pdus), b_var_gauss_high)

anew_mean_linear_low, anew_mean_linear_high = bootstrap.ci(data=(anew_lingress_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)
anew_mean_logistic_low, anew_mean_logistic_high = bootstrap.ci(data=(anew_logress_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)
anew_mean_gauss_low, anew_mean_gauss_high = bootstrap.ci(data=(anew_gauss_pred_pdus-a_pdus)**2, statfunction=np.mean, alpha=0.05)

print()
print("Training on Full HD and Testing on HDR")
print("-"*25, "Bootstrap | (CI_low, MSE, CI_high)", "-"*25)
print("-"*40, "HDR", "-"*40)
print("LINEAR MODEL")
print("MSE:", anew_mean_linear_low, mse(anew_lingress_pred_pdus, a_pdus), anew_mean_linear_high)

print("LOGISTIC MODEL")
print("MSE:", anew_mean_logistic_low, mse(anew_logress_pred_pdus, a_pdus), anew_mean_logistic_high)

print("GAUSSIAN MODEL")
print("MSE:", anew_mean_gauss_low, mse(anew_gauss_pred_pdus, a_pdus), anew_mean_gauss_high)

#plt.show()
