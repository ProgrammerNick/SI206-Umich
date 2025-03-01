# Import packages
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Define a path for import and export
path = 'C:/Users/Nicolas Newberry/OneDrive/Documents/Umich Assignments/fin427/'

# Import and view data
returns01 = pd.read_csv(path + 'Aggregate data 20250120_1144.csv')

print(returns01.head(10))
print(returns01.columns)

# Compile descriptive statistics
# For continuous variables show the mean, standard deviation and percentiles
# For indicator (1,0) variables show the mean
mean        =  returns01[['ret', 'lag1mcreal', 'adjret', 'gsret', 'indadjret', 'lnlag1mcreal', 'lag1bm', 'adjlag1bm','finlag1bm']].mean()
stddev      =  returns01[['ret', 'lag1mcreal', 'adjret', 'gsret', 'indadjret', 'lnlag1mcreal', 'lag1bm', 'adjlag1bm','finlag1bm']].std()
count       =  returns01[['ret', 'lag1mcreal', 'adjret', 'gsret', 'indadjret', 'lnlag1mcreal', 'lag1bm', 'adjlag1bm','finlag1bm']].count()
percentiles = (returns01[['ret', 'lag1mcreal', 'adjret', 'gsret', 'indadjret', 'lnlag1mcreal', 'lag1bm', 'adjlag1bm','finlag1bm']]
               .quantile([0, 0.125, 0.500, 0.875, 1]))
meanind     = returns01[['finlag1bmmiss']].mean()
print(mean)
print(stddev)
print(count)
print(percentiles)
print(meanind)

# What are the correlations between variables
correlation_matrix01 = returns01[['adjret','gsret','indadjret',  'lnlag1mcreal',  'adjlag1bm','finlag1bm','finlag1bmmiss']].corr()
correlation_matrix02 = returns01[['adjret','gsret','indadjret', 'zlnlag1mcreal', 'zadjlag1bm','finlag1bm','finlag1bmmiss']].corr()
correlation_matrix03 = returns01[['adjret','gsret','indadjret', 'rlnlag1mcreal', 'radjlag1bm','finlag1bm','finlag1bmmiss']].corr()
print(correlation_matrix01)
print(correlation_matrix02)
print(correlation_matrix03)

# Regression using statsmodels
y = returns01['indadjret']
x = returns01[['lnlag1mcreal','finlag1bm','finlag1bmmiss']]

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
print_model = model.summary()
b_coef = model.params
b_err = model.bse

# Calculating predicted values and Cook's D influence statistic, and merging with original dataset
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
analysis = pd.DataFrame({'predictions': predictions, 'cooks_d': cooks_d})
analysis = analysis.sort_values(by='cooks_d', ascending=False)

print(print_model)
print(f'R-squared: {model.rsquared:.5f}')
print(b_coef)
print(b_err)
print(analysis.columns)
print(analysis)

# Computations to understand what R-squared means
y_pred = model.predict(x)
ssr = np.sum((y - y_pred)**2)
sst = np.sum((y - np.mean(y))**2)
rsq = 1 - (ssr/sst)
rmse = np.sqrt(np.mean((y - y_pred)**2))

print(f'Sum of squared difference between y values and predicted y values (SSR): {ssr:.5f}')
print(f'Sum of squared difference between y values and average y values (SST): {sst:.5f}')
print(f'R-squared = 1 - SSR/SST: {rsq:.5f}')
print(f'Square root of the mean squared error: {rmse:.5f}')

# Export to Excel
with pd.ExcelWriter(path + 'Excel 02 Descriptive and OLS 20250120_1150.xlsx') as writer:
    mean.to_excel(writer, sheet_name='mn')
    stddev.to_excel(writer, sheet_name='sd')
    count.to_excel(writer, sheet_name='n')
    percentiles.to_excel(writer, sheet_name='perc')
    meanind.to_excel(writer, sheet_name='mnind')
    correlation_matrix01.to_excel(writer, sheet_name='corr01')
    correlation_matrix02.to_excel(writer, sheet_name='corr02')
    correlation_matrix03.to_excel(writer, sheet_name='corr03')
    b_coef.to_excel(writer, sheet_name='b_coef')
    b_err.to_excel(writer, sheet_name='b_err')

