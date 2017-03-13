import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.linear_model import LinearRegression
'''
I dati rappresentano
X = incendi per 1000 unita abitative
Y = furti per 1000 abitanti
all interno dello stesso CAP nell area metropolitana di Chicago
Riferimento Stati Uniti Commissione sui diritti civili
'''
    
#Valori X e Y
array_X = np.array([6.2, 9.5, 10.5, 7.7, 8.6, 34.1, 11, 6.9, 7.3, 15.1, 29.1, 2.2, 5.7, 2, 2.5, 4, 5.4, 2.2, 7.2, 15.1, 16.5, 18.4, 36.2, 39.7, 18.5, 23.3, 12.2, 5.6, 21.8, 21.6, 9, 3.6, 5, 28.6, 17.4, 11.3, 3.4, 11.9, 10.5, 10.7, 10.8, 4.8])
array_Y = np.array([29, 44, 36, 37, 53, 68, 75, 18, 31, 25, 34, 14, 11, 11, 22, 16, 27, 9, 29, 30, 40, 32, 41, 147, 22, 29, 46, 23, 4, 31, 39, 15, 32, 27, 32, 34, 17, 46, 42, 43, 34, 19])


#reshape necessario per il regressore
X = array_X.reshape(-1,1)
Y = array_Y.reshape(-1,1)

regr = LinearRegression() 
regr.fit(X, Y)

m = regr.coef_
q = regr.intercept_ 
covarianza= np.cov(array_X,array_Y)
coefPears = st.pearsonr(array_X,array_Y)

plt.figure()
plt.xlabel('X = incendi per 1000 unita abitative')
plt.ylabel('Y = furti per 1000 abitanti')
plt.title("m {0} - q {1}\nCoeff_Pearson {2}\n Covarianza {3}".format(m, q, coefPears[0],covarianza[0,1]))
plt.plot(X, Y, 'bx')
plt.plot(X, regr.predict(X), color='Red', linewidth=2)
plt.show()

print "DATI\n"
print "Covarianza\n" + str(covarianza[0,1]) + "\n"
print "Coefficiente di Pearson \n" + str(coefPears[0]) + "\n"
print "Coefficiente angolare (m)\n" + str(m) + "\n"
print "Termine noto (q)\n" + str(q)+ "\n"