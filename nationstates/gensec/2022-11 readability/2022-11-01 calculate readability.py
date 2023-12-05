# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:13:17 2022
@author: Imperium Anglorum
"""
import io

import pandas as pd
import scipy
import seaborn as sns

import textstat
import statsmodels.formula.api as smf

ops = pd.read_excel(
    '2022-10-30 gensec opinion readability statistics.xlsx',
    sheet_name='corpus')
ops.rename(columns=lambda s: s.strip().lower().replace(' ', '_'), inplace=True)

ops['text'] = ops['text'] \
    .str.replace(r'(?<=[^\n])\[\d+\]', '', regex=True) \
    .str.replace(r'@\.?', '', regex=True) \
    .str.strip()
ops = ops[ops['text'].notnull()]

ops['fkgl'] = ops['text'].apply(textstat.flesch_kincaid_grade)
ops['gfgl'] = ops['text'].apply(textstat.gunning_fog)
ops['smog'] = ops['text'].apply(textstat.smog_index)
ops['auto'] = ops['text'].apply(textstat.automated_readability_index)
ops['lins'] = ops['text'].apply(textstat.linsear_write_formula)

model = smf.rlm(
    formula=(
        'fkgl '
        '~ C(decision_author, Treatment("Sierra Lyricalia")) '
        '+ C(decision, Treatment("Illegal"))'
    ), data=ops).fit()

# model.resid.plot(kind='hist')
'''
Residuals seem pretty much normal. That's a positive.
'''

print(model.summary2(
    xname=[
        'Intercept',
        'author[Bananaistan]',
        'author[Christian Democrats]',
        'author[Glen-Rhodes]',
        'author[Imperium Anglorum]',
        'author[Sciongrad]',
        'author[Separatist Peoples]',
        'decision[Legal]'
    ]))

'''
                       Results: Robust linear model
===========================================================================
Model:                    RLM                    Df Residuals:       38    
Dependent Variable:       fkgl                   Norm:               HuberT
Date:                     2022-11-01 19:02       Scale Est.:         mad   
No. Observations:         46                     Cov. Type:          H1    
Df Model:                 7                      Scale:              1.5559
---------------------------------------------------------------------------
                             Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
---------------------------------------------------------------------------
Intercept                   13.2606   0.5446 24.3485 0.0000 12.1932 14.3280
author[Bananaistan]          0.1738   0.8762  0.1984 0.8427 -1.5434  1.8911
author[Christian Democrats] -0.9606   1.3143 -0.7309 0.4648 -3.5365  1.6153
author[Glen-Rhodes]          2.6391   1.5847  1.6653 0.0959 -0.4669  5.7450
author[Imperium Anglorum]   -1.0939   1.3143 -0.8323 0.4052 -3.6698  1.4820
author[Sciongrad]            1.6641   1.1993  1.3875 0.1653 -0.6865  4.0146
author[Separatist Peoples]   0.9115   0.8914  1.0226 0.3065 -0.8356  2.6585
decision[Legal]             -1.8993   0.7674 -2.4751 0.0133 -3.4033 -0.3953
===========================================================================
'''

model_gf = smf.rlm(
    formula=(
        'gfgl '
        '~ C(decision_author, Treatment("Sierra Lyricalia")) '
        '+ C(decision, Treatment("Illegal"))'
    ), data=ops).fit()

# model_gf.resid.plot(kind='hist')
'''
Residuals seem pretty much normal. That's a positive.
'''

print(model_gf.summary2(
    xname=[
        'Intercept',
        'author[Bananaistan]',
        'author[Christian Democrats]',
        'author[Glen-Rhodes]',
        'author[Imperium Anglorum]',
        'author[Sciongrad]',
        'author[Separatist Peoples]',
        'decision[Legal]'
    ]))

'''
                       Results: Robust linear model
===========================================================================
Model:                    RLM                    Df Residuals:       38    
Dependent Variable:       gfgl                   Norm:               HuberT
Date:                     2022-11-01 19:02       Scale Est.:         mad   
No. Observations:         46                     Cov. Type:          H1    
Df Model:                 7                      Scale:              1.7935
---------------------------------------------------------------------------
                             Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
---------------------------------------------------------------------------
Intercept                   14.3725   0.4784 30.0398 0.0000 13.4348 15.3103
author[Bananaistan]         -0.4000   0.7697 -0.5196 0.6033 -1.9086  1.1087
author[Christian Democrats] -1.0758   1.1546 -0.9318 0.3514 -3.3388  1.1871
author[Glen-Rhodes]          1.3606   1.3922  0.9773 0.3284 -1.3680  4.0892
author[Imperium Anglorum]   -2.1658   1.1546 -1.8759 0.0607 -4.4288  0.0971
author[Sciongrad]           -0.0044   1.0536 -0.0042 0.9967 -2.0694  2.0606
author[Separatist Peoples]   0.8567   0.7831  1.0940 0.2740 -0.6781  2.3914
decision[Legal]             -1.9162   0.6741 -2.8425 0.0045 -3.2375 -0.5949
===========================================================================
'''

model_sm = smf.rlm(
    formula=(
        'smog ~ C(decision_author, Treatment("Sierra Lyricalia")) '
        '+ C(decision, Treatment("Illegal"))'
    ), data=ops).fit()

# model_sm.resid.plot(kind='hist')
'''
Residuals are non-normal.
'''

print(model_sm.summary2(
    xname=[
        'Intercept',
        'author[Bananaistan]',
        'author[Christian Democrats]',
        'author[Glen-Rhodes]',
        'author[Imperium Anglorum]',
        'author[Sciongrad]',
        'author[Separatist Peoples]',
        'decision[Legal]'
    ]))

'''
                       Results: Robust linear model
===========================================================================
Model:                    RLM                    Df Residuals:       38    
Dependent Variable:       smog                   Norm:               HuberT
Date:                     2022-11-01 19:05       Scale Est.:         mad   
No. Observations:         46                     Cov. Type:          H1    
Df Model:                 7                      Scale:              1.1820
---------------------------------------------------------------------------
                             Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
---------------------------------------------------------------------------
Intercept                   15.0490   0.4113 36.5897 0.0000 14.2429 15.8551
author[Bananaistan]         -0.4022   0.6617 -0.6079 0.5433 -1.6991  0.8947
author[Christian Democrats] -0.0824   0.9925 -0.0830 0.9339 -2.0277  1.8630
author[Glen-Rhodes]          2.1863   1.1968  1.8268 0.0677 -0.1593  4.5319
author[Imperium Anglorum]   -1.2824   0.9925 -1.2920 0.1964 -3.2277  0.6630
author[Sciongrad]            1.0363   0.9057  1.1442 0.2525 -0.7389  2.8114
author[Separatist Peoples]   0.6899   0.6731  1.0248 0.3054 -0.6295  2.0092
decision[Legal]             -1.3706   0.5795 -2.3652 0.0180 -2.5065 -0.2348
===========================================================================
'''

model_ln = smf.rlm(
    formula=(
        'lins ~ C(decision_author, Treatment("Sierra Lyricalia")) '
        '+ C(decision, Treatment("Illegal"))'
    ), data=ops).fit()

# model_ln.resid.plot(kind='hist')
'''
Residuals have pretty fat tails.
'''

print(model_ln.summary2(
    xname=[
        'Intercept',
        'author[Bananaistan]',
        'author[Christian Democrats]',
        'author[Glen-Rhodes]',
        'author[Imperium Anglorum]',
        'author[Sciongrad]',
        'author[Separatist Peoples]',
        'decision[Legal]'
    ]))

'''
                       Results: Robust linear model
===========================================================================
Model:                    RLM                    Df Residuals:       38    
Dependent Variable:       lins                   Norm:               HuberT
Date:                     2022-11-01 19:03       Scale Est.:         mad   
No. Observations:         46                     Cov. Type:          H1    
Df Model:                 7                      Scale:              3.4565
---------------------------------------------------------------------------
                             Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
---------------------------------------------------------------------------
Intercept                   14.0849   1.3275 10.6104 0.0000 11.4831 16.6867
author[Bananaistan]          1.9470   2.1356  0.9117 0.3619 -2.2388  6.1327
author[Christian Democrats]  0.5040   3.2034  0.1573 0.8750 -5.7746  6.7826
author[Glen-Rhodes]          8.6710   3.8626  2.2448 0.0248  1.1003 16.2416
author[Imperium Anglorum]   -2.5462   3.2034 -0.7948 0.4267 -8.8248  3.7325
author[Sciongrad]           -0.8186   2.9232 -0.2800 0.7794 -6.5480  4.9108
author[Separatist Peoples]   3.2550   2.1726  1.4982 0.1341 -1.0033  7.5132
decision[Legal]             -3.5118   1.8704 -1.8775 0.0604 -7.1777  0.1542
===========================================================================
'''

# charts
sns.set_style("whitegrid")

# gfgl vs fkgl
sns.pairplot(ops)











