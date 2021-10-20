# dataset taken from https://www.kaggle.com/ananaymital/us-used-cars-dataset


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# choose one type of vehicle as a sample becausae the dataset is over 9GB


####### Uncomment to create samples from the dataset based on model_name
'''
# import a sample of he dataset, all vehicles with the value 'Jeep' in the make_name column of the population
iter_csv = pd.read_csv('used_cars_data.csv', iterator=True, chunksize=500)
raw_jeep = pd.concat([chunk[chunk['make_name'] == 'Jeep'] for chunk in iter_csv])

print(raw_jeep.head())
# print(len(raw_jeep()))

# save the sample dataset into a csv for easy reference and as a checkpoint in the process

raw_jeep.to_csv('jeep.csv')
'''

# Driver code for the application

# Import the sample dataset into memory
raw_jeep = pd.read_csv('jeep.csv')
raw_jeep['pre_2000'] = raw_jeep['year'] < 2000


# Add a boolean column that states before 2000 is True and after 2000 is False
pre_2000 = list()
for i in raw_jeep['year']:
    

    if i < 2000:
        pre_2000.append(True)
    else:
        pre_2000.append(False)

raw_jeep['pre_2000'] = pre_2000

#pre_2000 = raw_jeep[raw_jeep['year'] < 2000]
#post_2000 = raw_jeep[raw_jeep['year'] >= 2000]

pd.set_option('display.max_columns', None)
print(raw_jeep.head())



# plot a distribution of the number of cars sold after 31-12-1999

sns.countplot(data=raw_jeep[raw_jeep['year'] >= 2000], x="year")
plt.title('Number of cars by year post 2000')
plt.xlabel('Year of Production')
plt.ylabel('# of cars')
plt.xticks(rotation= 90)
# plt.xlim(1990, 2021)
plt.savefig('post_2000.pdf')
plt.show()

# plot a distribution of the number of cars sold on or before 31-12-1999
sns.countplot(data=raw_jeep[raw_jeep['year'] < 2000], x="year")
plt.title('Number of cars by year pre 2000')
plt.xlabel('Year of Production')
plt.ylabel('# of cars')
plt.xticks(rotation= 90)
# plt.xlim(1990, 2021)
plt.savefig('pre_2000.pdf')
plt.show()


# Lets check the missing value percentages
pd. set_option('display.max_rows', None)
my_missing = raw_jeep.isnull().sum()/len(raw_jeep)*100

print(my_missing)


# remove all variables with more than 20% missing values or higher
variables = raw_jeep.columns
col_length = len(raw_jeep.columns)
variable = []

for i in range(0,col_length):
    if my_missing[i] <= 20: 
        variable.append(variables[i])

print('The remaining columns for analysis are:')
print(variable)

# create new dataframe with the remaining variables
jeep_refined = raw_jeep.copy(deep=False)
jeep_refined = jeep_refined[variable]



# Lets convert the int and float string variables to numbers datatypes


# convert back_legroom and front_legroom to float datatypes
counter = 0
for i in jeep_refined.iterrows():
    print(i)
    if counter >= 5:
        break
    counter += 1


# Lets impute some values that can be found to be true

######################################################################################

# back_legroom (8378 records missing)
empty = jeep_refined[jeep_refined['back_legroom'].isna()]
print('empty values in variable back_legroom')
print(len(empty))
print(empty.head())

# lets look at the percentages of missing data within this subset of data
my_missing = empty.isnull().sum()/len(empty)*100
print('percentage of missing variables using the back_legroom NaN records ')
print(my_missing)



# let begin to either impute values or remove records from the records from the variable back_legroom
# plot back_legroom to year on a scatter using model_name as hue

sns.scatterplot(x='year', y='back_legroom', data=jeep_refined, hue='model_name', y_jitter= True, x_jitter=True)
plt.title('legroom by year')
plt.xlabel('year')
plt.ylabel('back_legroom')
plt.xticks(rotation= 90)
sns.despine()
plt.show()

# get a list of unique values in the column model_name as this column has 100% completeness
#
my_make_names = empty['model_name'].unique()

print('list of unique values from the model_name column')
print(my_make_names)

# make a scatterplot of each model_type by year and back_legroom

for i in my_make_names:

    model_name = jeep_refined[jeep_refined['model_name'] == i]
   



    sns.scatterplot(x='year', y='back_legroom', data=model_name, hue='engine_displacement').grid()
    plt.title(i +' numbers')
    plt.show()

# Data to be inferred
# Grand Cherokee ---- 1994-1998: 35.7 - 1999-2001: 35.3 - 2002-2003: 35.1 - 2004: 35.3 - 2005-2010: 35.5 - 2011-2021: 38.6
# Cherokee ---- 1983 is 37in, 1995 is 38.5in, 1991-1994 is 35.3in
# Liberty ---- 38.8in
# Compass ---- 39.3
# Renegade ---- 35.1

################# use more visualisations to find more constants that can be used to impute

##############################################################################################

# body_type





###############################################################################################

# city_fuel_economy









#################################################################################################

# engine_cylinders







# engine_displacement



################################################################################################

# engine_type




#################################################################################################

# exterior_color



##################################################################################################




# lets have a look to find variables with a low variance and we will decide if we can remove them safely

jeep_variance = jeep_refined.var()
print('The variance of each variable in the jeep_refined dataset')
print(jeep_variance)

# ID variables can be removed without issue to regression analysis. [0, listing_id, sp_id]
# ]franchise_dealer, is_new, seller_rating, pre_2000] are all boolean values with very small variance. 
# IDEA these variables could be used in a logistic regression predicting one of these values
# Will regression work better without them??????

# remove the ID variables
jeep_refined.drop(['sp_id'], axis=1, inplace=True)
jeep_refined.drop(['listing_id'], axis=1, inplace=True)
jeep_refined = jeep_refined.drop(columns=jeep_refined.columns[0])



print('Columns without ID variables')
print(jeep_refined.columns)


#############

# vehicle power variables: engine_displacement,horsepower

# maintenance variables: mileage, highway_fuel_economy, city_fuel_economy

# time based variables: year, daysonmarket

# location based variables: latitude, longitude, dealer_zip

# price based variables: savings_amount, price





# change jeep refined into all numbers

jeep_numeric = jeep_refined[['engine_displacement','horsepower','mileage', 'highway_fuel_economy', 'city_fuel_economy','year', 'daysonmarket', 'latitude', 'longitude', 'dealer_zip', 'savings_amount', 'price']]

# check pecentages of missing data in jeep_refined

my_missing = jeep_numeric.isnull().sum()/len(jeep_numeric)*100
print('All numbers missing data percentages')
print(my_missing)




################################################################################################

# the variable price is our Y variable, lets remove it from our dataset into its own variable y
y = jeep_refined['price']
print('dropping the price variable')
jeep_refined.drop(['price'], axis=1, inplace=True)

# variable without the price(y) variable
print('the columns without price variable')
print(jeep_refined.columns)
print('The dataset without the price variable')
print(jeep_refined.head())



# Run test_train_split on the data



# Now lets run a PCA analysis to see how many variables hold the most predictive power against the price variable

