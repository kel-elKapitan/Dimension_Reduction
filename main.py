# dataset taken from https://www.kaggle.com/ananaymital/us-used-cars-dataset


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path




def main():

    # choose one type of vehicle as a sample becausae the dataset is over 9GB

    # if dataset sample hasnt been saved, save file
    '''
    my_file = Path("input/jeep.csv")
    if my_file.is_file():
        # file exists
        print('File Exists')
        drive()
    '''
    # create a variable with all the unique values in the make_name
    
    def get_data(i):


        raw_test = []
        print('Creating csv for ' + i)
        # import samples of the dataset, all vehicles split with the make_name column
        iter_csv = pd.read_csv('used_cars_data.csv', iterator=True, chunksize=500)
        raw = pd.concat([chunk[chunk['make_name'] == i] for chunk in iter_csv])
        
        raw.to_csv('input/' + i + '.csv')
        print(i + ' descriptive stats')
        print(raw.describe())
        raw.describe().to_csv('output/' + i + '_Desc.csv')
        
          

        # print(len(raw_jeep()))

        # save the sample dataset into a csv for easy reference and as a checkpoint in the process

        print(i + '.csv saved')

        return

    # Create Datasets from the make_name column
    get_data('Jeep')
    get_data('Subaru')
    get_data('Mazda')
    get_data('Alfa Romeo')
    get_data('Land Rover')
    get_data('BMW')
    get_data('Hyundai')
    #get_data('Chevrolet')
    get_data('Lexus')
    get_data('Cadilac')
    get_data('Chrysler')
    get_data('Dodge')
    get_data('Mercedes Benz')
    get_data('Nissan')
    get_data('Honda')
    get_data('Kia')
    get_data('Lincoln')
    get_data('Audi')
    get_data('Jaguar')
    get_data('RAM')
    get_data('Volkswagon')
    get_data('Porsche')
    get_data('INFINITI')
    get_data('GMC')
    get_data('Acura')
    get_data('Toyota')
    get_data('Maserati')
    get_data('Volvo')
    










    drive()


    return 'The end of main!!!!!!!'


def drive():
    # Driver code for the application

    # Import the sample dataset into memory
    raw_jeep = pd.read_csv('input/Jeep.csv')
    raw_jeep['pre_2000'] = raw_jeep['year'] < 2000 # add bool column 

    # a quick look at the data
    pd.set_option('display.max_columns', None)
    #print(raw_jeep.head())

    # to see the values in a visually improved view, I have split the following 2 countplots into pre and post 2000

    # plot a distribution of the number of cars sold after 31-12-1999
    sns.countplot(data=raw_jeep[raw_jeep['year'] >= 2000], x="year")
    plt.title('Number of cars by year post 2000')
    plt.xlabel('Year of Production')
    plt.ylabel('# of cars')
    plt.xticks(rotation= 90)
    plt.savefig('output/post_2000.jpg')
    #plt.show()

    # plot a distribution of the number of cars sold on or before 31-12-1999
    sns.countplot(data=raw_jeep[raw_jeep['year'] < 2000], x="year")
    plt.title('Number of cars by year pre 2000')
    plt.xlabel('Year of Production')
    plt.ylabel('# of cars')
    plt.xticks(rotation= 90)
    plt.savefig('output/pre_2000.jpg')
    #plt.show()


    # Lets check the missing value percentages
    pd.set_option('display.max_rows', None)
    my_missing_1 = raw_jeep.isnull().sum()/len(raw_jeep)*100
    my_missing_1.to_csv('output/my_missing_percent_1.csv')
    #print(my_missing_1)

    # remove all variables with more than 20% missing values or higher
    variables = raw_jeep.columns
    col_length = len(raw_jeep.columns)
    variable = []


    for i in range(0,col_length):
        if my_missing_1[i] <= 20: 
            variable.append(variables[i])

    #print('The remaining columns for analysis are:')
    #print(variable)

    # create a shallow copy of new dataframe with the remaining variables
    jeep_refined = raw_jeep.copy(deep=False)
    jeep_refined = jeep_refined[variable]



    # Remove all records with a missing value

    jeep_refined = jeep_refined.dropna()

    #print(jeep_refined['fuel_tank_volume'].unique())

    my_missing = jeep_refined.isnull().sum()/len(jeep_refined)*100
    print('Percentage of missing values in variables after dropna()')
    print(my_missing)

    print('The length of the dataset after missing values have been removed is: ' + str(len(jeep_refined)))
    print('The number of the variables in the dataset after missing values have been removed is: ' + str(len(jeep_refined.columns)))


    # Lets convert the int and float string variables to numbers datatypes


    # convert back_legroom and front_legroom and fuel_tank_volume to float datatypes
    jeep_refined = jeep_refined.dropna()

    my_back_legroom = []
    my_front_legroom = []
    my_fuel_tank_volume = []

    for i in jeep_refined.iterrows():
        back_legroom = i[1][2]
        my_int = back_legroom.split(' ')
        my_int = float(my_int[0])
        my_back_legroom.append(my_int)

        front_legroom = i[1][15]
        my_int2 = front_legroom.split(' ')
        my_int2 = float(my_int2[0])
        my_front_legroom.append(my_int2)


        fuel_tank_volume = i[1][16]
        my_int3 = fuel_tank_volume.split(' ')
        my_int3 = my_int3[0]
        my_fuel_tank_volume.append(my_int3)

    jeep_refined['back_legroom'] = my_back_legroom
    jeep_refined['front_legroom'] = my_front_legroom
    jeep_refined['fuel_tank_volume'] = my_fuel_tank_volume




    # convert height, width and length to float

    jeep_refined['fuel_tank_volume'] = my_fuel_tank_volume
    jeep_refined['fuel_tank_volume'] = pd.to_numeric(jeep_refined['fuel_tank_volume'], downcast="float", errors='coerce')


    jeep_refined['maximum_seating'] = pd.to_numeric(jeep_refined['maximum_seating'], errors='coerce')
    jeep_refined['power'] = pd.to_numeric(jeep_refined['power'], errors='coerce')
    jeep_refined['torque'] = pd.to_numeric(jeep_refined['torque'], errors='coerce')
    jeep_refined['wheelbase'] = pd.to_numeric(jeep_refined['wheelbase'], downcast="float", errors='coerce')




    # Lets check the datatypes of the variables

    print(jeep_refined.dtypes)


# For speed I will not be imputing values as the dataset has plenty of samples amongst the population
# below is a list of vectors to impute instead of removing
######################################################################################

# back_legroom (8378 records missing)

##############################################################################################

# body_type

###############################################################################################

# city_fuel_economy

#################################################################################################

# engine_cylinders

#################################################################################################

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
    jeep_refined.drop(['vin'], axis=1, inplace=True)
    jeep_refined = jeep_refined.drop(columns=jeep_refined.columns[0])


    print('Columns without ID variables')
    print(jeep_refined.columns)


#############
#Variables grouped into there groups
# vehicle power variables: engine_displacement,horsepower

# maintenance variables: mileage, highway_fuel_economy, city_fuel_economy

# time based variables: year, daysonmarket

# location based variables: latitude, longitude, dealer_zip

# price based variables: savings_amount, price




    # Save jeep_refined into csv file
    jeep_refined.to_csv('output/jeep_refined.csv')

    # change jeep refined into all numbers
    jeep_numeric = jeep_refined[['back_legroom','city_fuel_economy','daysonmarket','dealer_zip','engine_displacement','franchise_dealer','front_legroom','highway_fuel_economy','horsepower','is_new','latitude','longitude','mileage','price','savings_amount','seller_rating','year','pre_2000']]

    # check pecentages of missing data in jeep_refined
    my_missing_2 = jeep_numeric.isnull().sum()/len(jeep_numeric)*100
    my_missing_2.to_csv('output/my_missing_percent_2.csv')

    print('All numbers missing data percentages')
    print(my_missing_2)





    
    ################################################################################################

    # the variable price is our Y variable, lets remove it from our dataset into its own variable y
    y = jeep_numeric['price']

    # Save the y variable in a csv file
    y.to_csv('output/y.csv')


    #print('dropping the price variable')
    #jeep_numeric.drop(['price'], axis=1, inplace=True)

    # variable without the price(y) variable
    #print('the columns without price variable')
    #print(jeep_numeric.columns)
    #print('The dataset without the price variable')
    #print(jeep_numeric.head())

    # Save jeep_numeric to csv file
    jeep_numeric.to_csv('output/jeep_numeric.csv')



    # lets look for correlation between the remaining variables




    sns.heatmap(jeep_numeric.corr(), annot=True)
    plt.savefig('output/correlation.pdf')
    #plt.show()
    # Lets check the datatypes of the variables

    print('datatypes of the remaining variables')
    print(jeep_numeric.dtypes)








# Now lets run a PCA analysis to see how many variables hold the most predictive power against the price variable





    pca=PCA()

    jeep_tf = pca.fit_transform(jeep_numeric)
    # plot the variance
    plt.plot(pca.explained_variance_ratio_)
    plt.savefig('output/PCA_var_ratio.jpg')
    #plt.show()



    counter = 0
    my_max = len(jeep_refined['model_name'].unique()) - 1
    # plot shows we can safely drop the number of x variables to 2
    for i in jeep_refined['model_name'].unique():
        counter += 1
        if counter == my_max:
            break

        my_model = jeep_numeric[jeep_refined['model_name'] == i]
        pca = PCA(n_components= 3)
        pca.fit(my_model)
        my_pca = pca.transform(my_model)

# visualise pca output
    
        sns.scatterplot(x=my_model['price'], y=my_pca[:,0], hue=my_model['year'])
        plt.title('PCA with only ' + i)
        plt.savefig('output/' + i + 'PCA_year.jpg')
        #plt.show()


    return 'End of drive'







# Run test_train_split on the data





main()

