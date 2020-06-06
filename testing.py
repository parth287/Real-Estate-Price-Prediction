from exp import housing_test, pipe, model, housing_test_label, using_model
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error

# pipe.fit(housing_test)
# x_prepared=pipe.transform(housing_test)
# housing_test =pd.DataFrame(x_prepared, columns = housing_test.columns)
# housing_test_tr = pipe.fit_transform(housing_test)
# pred = model.predict(housing_test_tr)
# f_mse = mean_squared_error(housing_test_label, pred)
# f_rsme = np.sqrt(f_mse)
# print("The mean square error is : ",f_mse)
# print("The root of mean square error is : ",f_rsme)
# print(pred, list(housing_test_label))


# SUng the func made in exp
using_model(housing_test, housing_test_label)