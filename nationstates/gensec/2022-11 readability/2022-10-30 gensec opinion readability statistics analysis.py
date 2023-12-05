# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:15:36 2022
@author: Imperium Anglorum
"""
import scipy
import io
import pandas as pd
import statsmodels.formula.api as smf

s = '''Challenged proposal	Citation	Sig	Decision	Decision author	FKGL
Repeal "Ban on Secret Treaties"	[2018] GAS 6		Illegal	Sierra Lyricalia	10.5
Safeguarding Nuclear Materials	[2017] GAS 12		Legal	Separatist Peoples	11.0
Repeal "Clean Prostitute Act"	[2021] GAS 5		Illegal	Sierra Lyricalia	11.3
Freedom of Opinion and Belief	[2021] GAS 8		Illegal	Bananaistan	11.4
Pandemic Waste Awareness Act	[2021] GAS 6		Illegal	Sierra Lyricalia	11.8
Repeal "On Universal Jurisdiction"	[2018] GAS 8		Illegal	Bananaistan	12.1
Preventing Desertification	[2018] GAS 5	P	Legal	Sierra Lyricalia	12.3
Repeal "On Scientific Cooperation" PART 1	[2022] GAS 2		Illegal	Imperium Anglorum	12.3
WA Peacekeeping Charter	[2016] GAS 4		Illegal	Christian Democrats	12.6
Repeal "Pesticide Regulations"	[2017] GAS 7		Illegal	Sierra Lyricalia	12.7
Administrative Compliance Act	[2018] GAS 7		Legal	Sierra Lyricalia	13.1
International Art Gallery	[2022] GAS 4		Illegal	Imperium Anglorum	13.1
Repeal "Safety and Integrity in Conflict Journalism"	[2022] GAS 1		Illegal	Imperium Anglorum	13.6
Protection of Nuclear Armaments	[2017] GAS 1		Legal	Separatist Peoples	13.8
Repeal "Ban on the Sterilisation of Minors etc"	[2019] GAS 2		Illegal	Separatist Peoples	13.9
Repeal "On Scientific Cooperation" PART 2	[2022] GAS 2		Illegal	Sierra Lyricalia	13.9
Repeal "Ban on Secret Treaties"	[2019] GAS 3		Illegal	Sierra Lyricalia	14.1
Repeal "Responsibility in Transferring Arms"	[2017] GAS 8		Illegal	Sierra Lyricalia	14.4
Ban on Secret Treaties	[2017] GAS 10	N	Illegal	Sierra Lyricalia	14.4
Extrajudicial Punishment Ban	[2017] GAS 6		Legal	Sciongrad	14.9
Agricultural Invasive Species Act	[2018] GAS 4		Legal	Separatist Peoples	14.9
Oceanic Hazardous Waste Disposal Ban	[2020] GAS 1 bis		Illegal	Sierra Lyricalia	14.9
Trade of Endangered Organisms	[2016] GAS 6	N	Illegal	Sierra Lyricalia	15.1
International Aviation Act	[2016] GAS 3		Illegal	Christian Democrats	15.5
Repeal "Ban on the Sterilisation of Minors etc"	[2020] GAS 3		Legal	Separatist Peoples	15.6
Repeal "Epidemic Investigation Act"	[2022] GAS 3		Illegal	Sierra Lyricalia	15.9
Compliance Commission	[2016] GAS 5		Legal	Glen-Rhodes	16.1
Promoting Research on Life in Foetuses and Embryos	[2018] GAS 3	P	Illegal	Christian Democrats	16.1
Freedom to Seek Care	[2017] GAS 11		Illegal	Bananaistan	16.2
Patent Recognition Treaty	[2017] GAS 4		Legal	Separatist Peoples	16.4
Repeal "Freedom to Seek Medical Care"	[2018] GAS 1		Legal	Bananaistan	16.7
Repeal "Protection of Biomedical Research"	[2018] GAS 2		Legal	Bananaistan	17.1
The International Immigration Standard	[2017] GAS 3		Illegal	Sciongrad	17.8
Standards on Police Accountability	[2020] GAS 2		Illegal	Sciongrad	18.0
National Control of Elections	[2016] GAS 2		Legal	Sciongrad	18.1
Repeal "On Universal Jurisdiction"	[2018] GAS 9		Illegal	Sierra Lyricalia	18.5
Repeal "Rights of the Quarantined"	[2017] GAS 9		Illegal	Separatist Peoples	19.3
Protections During Territorial Transitions	[2021] GAS 4		Illegal	Bananaistan	19.7
Ban on the Involuntary Administration of Drugs	[2020] GAS 1		Illegal	Bananaistan	20.2
Repeal "Nuclear Arms Possession Act"	[2016] GAS 1		Illegal	Separatist Peoples	21.8
Resolution 393	[2017] GAS 2		Illegal	Glen-Rhodes	22.1
Repeal "Digital Network Defense"	[2021] GAS 1		Illegal	Bananaistan	23.3
Framework For Proper Financial Reporting Standards	[2021] GAS 2		Illegal	Sierra Lyricalia	26.1
The Cloning Conventions	[2019] GAS 1		Illegal	Separatist Peoples	28.0
Repeal "Preventing the Execution of Innocents"	[2018] GAS 10		Illegal	Bananaistan	31.4
Repeal "Rights of Crime Victims"	[2021] GAS 7		Illegal	Bananaistan	31.9
'''
ops = pd.read_csv(io.StringIO(s), sep='\t')
ops = pd.concat([
    ops,
    pd.get_dummies(ops['Decision author']).rename(
        columns=lambda x: 'Author__' + str(x)),
    pd.get_dummies(ops['Decision'])
], axis=1)
ops.rename(
    columns=lambda s: s.strip().lower().replace(' ', '_').replace('-', ''),
    inplace=True)

model = smf.rlm(
    formula='fkgl ~ author__bananaistan + author__christian_democrats + author__glenrhodes + author__sciongrad + author__separatist_peoples + author__imperium_anglorum + legal',
    data=ops).fit()
print(model.summary())

model = smf.ols(
    formula='fkgl ~ author__bananaistan + author__christian_democrats + author__glenrhodes + author__sciongrad + author__separatist_peoples + author__imperium_anglorum + legal',
    data=ops).fit()
print(model.summary())

"""
                    Robust linear Model Regression Results                    
==============================================================================
Dep. Variable:                   fkgl   No. Observations:                   46
Model:                            RLM   Df Residuals:                       38
Method:                          IRLS   Df Model:                            7
Norm:                          HuberT                                         
Scale Est.:                       mad                                         
Cov Type:                          H1                                         
Date:                Mon, 31 Oct 2022                                         
Time:                        00:47:34                                         
No. Iterations:                    26                                         
===============================================================================================
                                  coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                      14.4565      0.805     17.963      0.000      12.879      16.034
author__bananaistan             5.5738      1.251      4.454      0.000       3.121       8.027
author__christian_democrats     0.2769      1.941      0.143      0.887      -3.528       4.082
author__glenrhodes              6.2488      2.342      2.669      0.008       1.659      10.838
author__sciongrad               4.3488      1.773      2.453      0.014       0.874       7.824
author__separatist_peoples      4.1120      1.379      2.982      0.003       1.410       6.814
author__imperium_anglorum      -1.4565      1.941     -0.750      0.453      -5.261       2.348
legal                          -3.2106      1.152     -2.787      0.005      -5.468      -0.953
===============================================================================================
"""

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   fkgl   R-squared:                       0.315
Model:                            OLS   Adj. R-squared:                  0.189
Method:                 Least Squares   F-statistic:                     2.500
Date:                Mon, 31 Oct 2022   Prob (F-statistic):             0.0324
Time:                        00:47:34   Log-Likelihood:                -130.32
No. Observations:                  46   AIC:                             276.6
Df Residuals:                      38   BIC:                             291.3
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                      15.1413      1.190     12.722      0.000      12.732      17.551
author__bananaistan             5.6707      1.851      3.064      0.004       1.924       9.417
author__christian_democrats    -0.4080      2.871     -0.142      0.888      -6.219       5.403
author__glenrhodes              5.9886      3.463      1.729      0.092      -1.022      12.999
author__sciongrad               4.0886      2.622      1.560      0.127      -1.219       9.396
author__separatist_peoples      4.3030      2.039      2.111      0.041       0.176       8.430
author__imperium_anglorum      -2.1413      2.871     -0.746      0.460      -7.953       3.670
legal                          -4.0598      1.703     -2.383      0.022      -7.508      -0.611
==============================================================================
Omnibus:                       10.264   Durbin-Watson:                   0.607
Prob(Omnibus):                  0.006   Jarque-Bera (JB):               11.419
Skew:                           0.787   Prob(JB):                      0.00331
Kurtosis:                       4.866   Cond. No.                         6.56
==============================================================================
"""

scipy.stats.ttest_ind(
    ops.loc[ops['legal'] == 1, 'fkgl'],
    ops.loc[ops['illegal'] == 1, 'fkgl'],
    equal_var=False)
# Ttest_indResult(statistic=-1.538583472456826, pvalue=0.1434567917367788)
