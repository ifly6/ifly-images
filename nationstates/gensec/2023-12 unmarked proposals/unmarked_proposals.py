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
merged_data['month'] = merged_data['creation_date'].round('D') \
    + pd.offsets.Week(weekday=6)
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
    smf.ols('id_det ~ chamber + chamber:after_5', data=unmarked)
    .fit(cov_type='HC3').summary2()
)

"""
                 Results: Ordinary least squares
==================================================================
Model:              OLS              Adj. R-squared:     -0.005   
Dependent Variable: id_det           AIC:                -134.6139
Date:               2023-12-05 22:18 BIC:                -122.4145
No. Observations:   156              Log-Likelihood:     71.307   
Df Model:           3                F-statistic:        1.335    
Df Residuals:       152              Prob (F-statistic): 0.265    
R-squared:          0.014            Scale:              0.024087 
------------------------------------------------------------------
                      Coef.  Std.Err.   z    P>|z|   [0.025 0.975]
------------------------------------------------------------------
Intercept             0.1850   0.0370 4.9992 0.0000  0.1124 0.2575
chamber[T.SC]         0.0630   0.0910 0.6926 0.4885 -0.1153 0.2413
chamber[GA]:after_5   0.0763   0.0420 1.8168 0.0692 -0.0060 0.1586
chamber[SC]:after_5   0.0165   0.0848 0.1942 0.8460 -0.1497 0.1827
------------------------------------------------------------------
Omnibus:               10.581       Durbin-Watson:          1.520 
Prob(Omnibus):         0.005        Jarque-Bera (JB):       10.732
Skew:                  0.613        Prob(JB):               0.005 
Kurtosis:              3.383        Condition No.:          10    
==================================================================
Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
"""
