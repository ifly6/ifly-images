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
merged_data['month'] = merged_data['creation_date'].round('D') + pd.offsets.Week(weekday=6)
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

smf.ols('id_det ~ chamber', data=prop_unmarked.reset_index()) \
    .fit(cov_type='HC2').summary2()

"""
                 Results: Ordinary least squares
==================================================================
Model:              OLS              Adj. R-squared:     -0.005   
Dependent Variable: id_det           AIC:                -136.5604
Date:               2023-12-05 21:34 BIC:                -130.4607
No. Observations:   156              Log-Likelihood:     70.280   
Df Model:           1                F-statistic:        0.1646   
Df Residuals:       154              Prob (F-statistic): 0.686    
R-squared:          0.001            Scale:              0.024089 
-------------------------------------------------------------------
                 Coef.   Std.Err.     z     P>|z|    [0.025  0.975]
-------------------------------------------------------------------
Intercept        0.2525    0.0181  13.9738  0.0000   0.2171  0.2879
chamber[T.SC]    0.0101    0.0249   0.4057  0.6850  -0.0386  0.0588
------------------------------------------------------------------
Omnibus:               11.323       Durbin-Watson:          1.491 
Prob(Omnibus):         0.003        Jarque-Bera (JB):       11.607
Skew:                  0.629        Prob(JB):               0.003 
Kurtosis:              3.451        Condition No.:          3     
==================================================================
Notes:
[1] Standard Errors are heteroscedasticity robust (HC2)
"""
