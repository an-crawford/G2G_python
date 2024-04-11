#imports
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.optimize import minimize

# Background functions 

# create a logdiffexp function to avoid computation error
def logdiffexp(a, b):
    c = np.maximum(a, b)
    return c + np.log(np.exp(a - c) - np.exp(b - c))

# model log-likelihood function
def G2G_varying_LL(par, data_df):
    # model variables: 
    # par : parameters
    # data_df : a pandas data frame with columns: id, id of each subject, time, duration observed for each subject, status   / 
    #           binary event variable, name_not_sure, covariance matrix
    # par[0] = mean of BG
    # par[1] = polarization of BG

    r = par[0]
    alpha = par[1]
    coeff = par[2:]

    X = data_df.iloc[:, 3:].values

    data_df['Ct'] = np.exp(np.dot(X, coeff))

    data_df['cumsumCt'] = data_df.groupby('id')['Ct'].cumsum()
    # gather unique id information
    id_df = data_df.groupby('id').agg({'time': 'max', 'status': 'max'}).reset_index()
    # uncensored piece of likelihood
    LL_uncen = 0
    uncen_time = id_df.loc[id_df['status'] == 1, 'time'].unique()

    for t in uncen_time:
        id_need = id_df.loc[(id_df['status'] == 1) & (id_df['time'] == t), 'id'].values

        if t == 1:
            cumsumCt_b = data_df.loc[(data_df['id'].isin(id_need)) & (data_df['time'] == 1), 'cumsumCt']
            LL_uncen += np.sum(logdiffexp(0, -r * np.log(1 + cumsumCt_b / alpha)))
        else:
            cumsumCt_a = data_df.loc[(data_df['id'].isin(id_need)) & (data_df['time'] == (t - 1)), 'cumsumCt']
            cumsumCt_b = data_df.loc[(data_df['id'].isin(id_need)) & (data_df['time'] == t), 'cumsumCt']
            LL_uncen += np.sum(logdiffexp(-r * np.log(1 + cumsumCt_a / alpha), -r * np.log(1 + cumsumCt_b / alpha)))
    #censored piece of likelihood
    LL_cen = 0
    cen_time = id_df.loc[id_df['status'] == 0, 'time'].unique()

    for t in cen_time:
        id_need = id_df.loc[(id_df['status'] == 0) & (id_df['time'] == t), 'id'].values
        cumsumCt_b = data_df.loc[(data_df['id'].isin(id_need)) & (data_df['time'] == t), 'cumsumCt']
        LL_cen += np.sum((-r) * np.log(1 + cumsumCt_b / alpha))

    return -(LL_uncen + LL_cen)

# MLE 

# Data Prep
def G2G_varying_MLE(time, status, data, subject):
    # time: the time dependent variable
    # status: the status dependent variable
    # data: data frame
    # id: text field for the subject 

    # the dependent variables: time and status 
    time_name, status_name = time, status

    df_temp = data.drop(columns=[time_name, status_name], axis = 1)

    X = pd.get_dummies(df_temp, drop_first=True)

    model_data = pd.DataFrame({
        'id': data[subject].values.flatten(),
        'time': data[time_name].values.flatten(),
        'status': data[status_name].values.flatten(),
        'X': X.values.tolist()
    })

    return G2G_varying_optim(model_data)

# function used in the MLE function
'''def G2G_varying_optim(model_data):
    nvar = model_data.shape[1] - 1

    solution = minimize(
        lambda par: G2G_varying_LL(par, model_data),
        x0=np.concatenate(([0.5, 0.05], np.zeros(nvar - 2))),
        args=(model_data,),
        method='L-BFGS-B',
        bounds=[(0.001, None), (0.001, None)] + [(-5, 5)] * (nvar - 2),
        options={'maxiter': 1000},
    )

    par_stderr = np.sqrt(np.diag(np.linalg.inv(solution.hess_inv)))
    solution.par_stderr = par_stderr
    solution.par_upper = solution.x + 1.96 * par_stderr
    solution.par_lower = solution.x - 1.96 * par_stderr

    return solution
    '''


def G2G_varying_optim(model_data):
    nvar = model_data.shape[1] - 1
    
    # Define the objective function
    def objective_function(par):
        return G2G_varying_LL(par, model_data)
    
    # Set initial parameter values
    par = np.concatenate(([0.5, 0.05], np.zeros(nvar - 2)))
    
    # Set lower and upper bounds for parameters
    lower_bounds = [0.001, 0.001] + [-5.0] * (nvar - 2)
    upper_bounds = [np.inf, np.inf] + [5.0] * (nvar - 2)
    
    # Perform optimization
    solution = minimize(objective_function, x0=par, method='L-BFGS-B',
                        bounds=list(zip(lower_bounds, upper_bounds)), options={'maxiter': 1000})
    
    # Compute standard errors
    par_stderr = np.sqrt(np.diag(np.linalg.inv(solution.hess_inv)))
    
    # Compute upper and lower bounds for parameters
    par_upper = solution.x + 1.96 * par_stderr
    par_lower = solution.x - 1.96 * par_stderr
    
    # Return the optimization result along with standard errors and bounds
    return solution, par_stderr, par_upper, par_lower
