#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:22:42 2023
@author: Imperium Anglorum
"""
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

prop = pd.read_csv('proposals.csv.xz', parse_dates=['creation_date'])
decs = pd.read_csv('decisions.csv.xz', parse_dates=['time'])

merged_data = prop.merge(
    decs, left_on='id', right_on='proposal_id', how='left', suffixes=['_prt', '_det'])
merged_data['month'] = merged_data['creation_date'].round(
    'D') + pd.offsets.Week(weekday=6)
merged_data = merged_data[
    merged_data['creation_date'] > merged_data['time'].min()]

prop_unmarked = merged_data \
    .groupby(['chamber', 'month', 'id_prt'])['id_det'] \
    .count() \
    .groupby(level=[0, 1]) \
    .apply(lambda s: (s == 0).mean())

prop_unmarked.unstack(0).to_csv(
    'proposals unmarked by chamber.csv', index=True)

# ----------------------------------------------------------------------------
# plot ga and sc comparison

f, ax = plt.subplots(figsize=(8, 6))
prop_unmarked.unstack(0).plot(ax=ax)
ax.set_title('Proportion of proposals unmarked')
ax.set_ylabel('Proportion')

f.savefig('unmarked_proportion.jpeg', bbox_inches='tight')

# ----------------------------------------------------------------------------
# is the rate of marking proposals statistically distinguishable

unmarked = prop_unmarked.reset_index()

# after reducing to 5
unmarked['after_5'] = (unmarked['month'] >= '2022-08-20').astype(int)

# save
unmarked.to_csv(
    'proposals unmarked by chamber.csv', index=False)

print(
    smf.ols('id_det ~ chamber + after_5', data=unmarked)
    .fit(cov_type='HC3').summary2()
)

"""
                 Results: Ordinary least squares
==================================================================
Model:              OLS              Adj. R-squared:     -0.003   
Dependent Variable: id_det           AIC:                -136.0078
Date:               2023-12-05 21:58 BIC:                -126.8583
No. Observations:   156              Log-Likelihood:     71.004   
Df Model:           2                F-statistic:        0.8269   
Df Residuals:       153              Prob (F-statistic): 0.439    
R-squared:          0.010            Scale:              0.024022 
-------------------------------------------------------------------
                  Coef.   Std.Err.    z     P>|z|    [0.025  0.975]
-------------------------------------------------------------------
Intercept         0.2114    0.0409  5.1745  0.0000   0.1313  0.2915
chamber[T.SC]     0.0101    0.0248  0.4059  0.6848  -0.0386  0.0588
after_5           0.0464    0.0442  1.0501  0.2937  -0.0402  0.1330
------------------------------------------------------------------
Omnibus:               12.487       Durbin-Watson:          1.518 
Prob(Omnibus):         0.002        Jarque-Bera (JB):       13.044
Skew:                  0.669        Prob(JB):               0.001 
Kurtosis:              3.465        Condition No.:          6     
==================================================================
Notes:
[1] Standard Errors are heteroscedasticity robust (HC2)
"""

# ----------------------------------------------------------------------------
# rate of marking proposals statistically distinguishable, proposal level?

pm = merged_data.groupby('ns_id')['id_det'].count().reset_index()

# construct regressors
pm['unmarked'] = (pm['id_det'] == 0).astype(int)
pm['chamber'] = pm['ns_id'].map(prop.set_index('ns_id')['chamber'])
pm['creation_date'] = pm['ns_id'].map(
    prop.set_index('ns_id')['creation_date'])
pm['after_5'] = (pm['creation_date'] >= '2022-08-20').astype(int)

print(
    smf.logit('unmarked ~ chamber + after_5', data=pm)
    .fit().summary2()
)

"""
                         Results: Logit
================================================================
Model:              Logit            Method:           MLE      
Dependent Variable: unmarked         Pseudo R-squared: 0.002    
Date:               2023-12-05 22:04 AIC:              2945.1869
No. Observations:   2579             BIC:              2962.7523
Df Model:           2                Log-Likelihood:   -1469.6  
Df Residuals:       2576             LL-Null:          -1472.1  
Converged:          1.0000           LLR p-value:      0.082521 
No. Iterations:     5.0000           Scale:            1.0000   
----------------------------------------------------------------
                  Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
----------------------------------------------------------------
Intercept        -1.3895   0.1722 -8.0698 0.0000 -1.7270 -1.0521
chamber[T.SC]     0.1138   0.0939  1.2116 0.2257 -0.0703  0.2979
after_5           0.3157   0.1774  1.7798 0.0751 -0.0319  0.6633
================================================================
"""