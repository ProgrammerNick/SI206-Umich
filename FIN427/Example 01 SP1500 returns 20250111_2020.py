#Import and view data
import matplotlib.pyplot as plt
import pandas as pd

path = 'C:/Users/Nicolas Newberry/OneDrive/Documents/Umich Assignments/FIN427/'

SPTM_prices = pd.read_csv(path + 'SPTM prices 20250111.csv', index_col=0)
SPTM_dividends = pd.read_csv(path + 'SPTM dividends 20250111.csv', index_col=0)

print('SPTM prices:')
print(SPTM_prices)
print('SPTM dividends:')
print(SPTM_dividends)

#Merge datasets
combined = SPTM_prices.merge(SPTM_dividends, left_index=True, right_index=True, how='left')
print('Combined dataset:')
print(combined)

#Replace not a number dividends with zero
combined.fillna(0, inplace=True)
print('Combined dataset:')
print(combined)

#Compute capital gains, dividend yield, total returns and a total returns index
combined['lag1Price'] = combined['Price'].shift(1)
combined['cg']=combined['Price']/combined['lag1Price']-1
combined['dy']=combined['Dividends']/combined['lag1Price']
combined['tr']=combined['cg']+combined['dy']
print('Combined dataset:')
print(combined)
first_date = combined.index.min()
print(first_date)
initial_index_value = 1
combined['cumretfact'] = 1 + combined['tr']
combined['total_return_index'] = initial_index_value * combined['cumretfact'].cumprod()
print('Combined dataset:')
print(combined)

#Export to Excel
with pd.ExcelWriter('SP indices 20250111.xlsx') as writer:
    combined.to_excel(writer, sheet_name='sptm')

#Plot a chart
combined['total_return_index'].plot(label='SPTM', color='red')
plt.title('Cumulative Return to $1')
plt.xlabel('Date')
plt.legend()
plt.savefig(path + 'ETF returns 20250111.jpg')
plt.show()