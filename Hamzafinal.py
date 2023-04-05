import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import wbdata

def read_world_bank_data(file_path):
    """Reads the World Bank format file into a DataFrame.

    Args:
        file_path (str): The path to the file.

    Returns:
        pd.DataFrame: The DataFrame containing the data.
    """
    df = pd.read_csv(r'C:\Users\user\Desktop\MSc Data Science\Semester One\Applied Data Science\Assignment No 2\hamza\1\Dataset.csv')

    # Extract country dataframe
    df_countries = df.iloc[:, :4]
    df_countries.columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    df_countries.set_index('Country Name', inplace=True)

    # Extract year dataframe
    df_years = df.iloc[:, 4:].transpose()
    df_years.columns = df_countries.index
    df_years.index.name = 'Year'
    
    # Eliminate null values and clean the resulting DataFrames
    df_countries.dropna(inplace=True)
    df_years.dropna(inplace=True)

    return df, df_countries, df_years

# Create a dictionary with the data for the new indicators
new_indicators = {'Country': ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola'],
                  'Arable land (% of land area)': [11.98, 22.16, 3.14, 2.22, 3.31],
                  'Electricity consumption (kWh per capita)': [99.3, 3105, 1437, 8506, 190],
                  'Percentage of GDP': [23.9, 25.5, 35.3, 39.7, 33.2],
                  'Population': [38928346, 2877797, 43851044, 77265, 32866268]}

# Using Describe Function

def get_country_data(indicators, countries):
    """
    Returns sub-dataframes for each country in the given dictionary
    of countries and their ISO codes,
    containing data for the given indicators retrieved from the
    World Bank database.
    Parameters:
    indicators (dict): A dictionary containing the indicators
    to retrieve from the World Bank database.
    countries (dict): A dictionary containing the ISO codes
    and names of the countries to retrieve data for.

    Returns:
    dict: A dictionary containing sub-dataframes for each
    country, with the country name as the key.
    """
    # Load the data into a Pandas DataFrame.
    data = wbdata.get_dataframe(indicators, country=list(countries.keys()), convert_date=False)
    
    # Rename the columns and index.
    data.rename(columns={indicators["EG.ELC.ACCS.ZS"]: "Access to electricity", 
                         indicators["SP.POP.GROW"]: "Population growth"}, inplace=True)
    
    # Update the indicators dictionary to reflect the new names.
    indicators = {"EG.ELC.ACCS.ZS": "Electricity Access (% of population)", 
                  "SP.POP.GROW": "Population Growth (annual %)"}

    # Create sub-dataframes for each country.
    country_dfs = {}
    for code, name in countries.items():
        df = data.xs(name, level=0)
        country_dfs[name] = df

    # Print summary statistics for each sub-dataframe.
    for name, df in country_dfs.items():
        print(name + ":\n", df.describe())
        
    return country_dfs

indicators = {"EG.ELC.ACCS.ZS": "Access to electricity (% of population)", 
              "SP.POP.GROW": "Population growth (annual %)"}
countries = {"CN": "China", "IN": "India", "US": "United States", 
             "PK": "Pakistan", "BD": "Bangladesh", "ID": "Indonesia"}
country_dfs = get_country_data(indicators, countries)


#Using other Statictics Method


def calculate_mean(data):
    """
    Calculates the mean values for each numeric column in a dictionary.

    Parameters:
        data (dict): A dictionary containing column names as keys
        and a list of values as values.

    Returns:
        dict: A dictionary containing the mean values for each
        numeric column in the input dictionary.
    """
    mean_values = {}
    for key, values in data.items():
        if isinstance(values[0], (int, float)):
            mean_values[key] = round(statistics.mean(values), 2)
    return mean_values


def calculate_median(data):
    """Calculates the median of each column in a dictionary of data.

    Args:
        data (dict): A dictionary of data where each key
        corresponds to the name of a column and each value is a list of values for that column.

    Returns:
        dict: A dictionary where each key is the name of
        a column and each value is the median value of that column.

    """
    median_values = {}
    for key, values in data.items():
        if isinstance(values[0], (int, float)):
            median_values[key] = round(statistics.median(values), 2)
    return median_values


def calculate_mode(data):
    """
    Calculates the mode for each column in the given data.

    Parameters:
    data (dict): A dictionary with column names as keys and lists of values as values.

    Returns:
    dict: A dictionary with column names as keys and mode values as values.
    """
    mode_values = {}
    for key, values in data.items():
        if isinstance(values[0], (int, float)):
            mode_values[key] = statistics.mode(values)
    return mode_values

mean_values = calculate_mean(new_indicators)
median_values = calculate_median(new_indicators)
mode_values = calculate_mode(new_indicators)


print("Mean values:")
for key, value in mean_values.items():
    print(f"{key}: {value}")

print("\nMedian values:")
for key, value in median_values.items():
    print(f"{key}: {value}")

print("\nMode values:")
for key, value in mode_values.items():
    print(f"{key}: {value}")
       
    
    
# Select columns to plot and get mean values for them
selected_columns = ['Arable land (% of land area)', 
                    'Electricity consumption (kWh per capita)', 
                    'Percentage of GDP']
mean_values = {}
for column in selected_columns:
    mean_values[column] = round(statistics.mean(new_indicators[column]), 2)

# Get population values for all countries
population_values = new_indicators['Percentage of GDP']

# Plot histograms
fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey=True)
fig.subplots_adjust(wspace=0.3)

for i, column in enumerate(selected_columns):
    axs[i].hist(new_indicators[column], alpha=0.7, color='blue', edgecolor='white')
    axs[i].axvline(mean_values[column], color='red', linestyle='dashed', linewidth=1, label='Mean')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Frequency')
    axs[i].set_title(column)
    axs[i].legend()

plt.suptitle('Histograms of Mean Values for Selected Indicators')
plt.show()


# Manual data
population_values = [i for i in range(264)]
median_values = {
    'GDP per capita': [i*8 for i in range(264)],
    'Life expectancy': [60 + i*0.5 for i in range(264)],
    'Education': [i/10 for i in range(264)],
    'Health': [i/20 for i in range(264)],
    'Environment': [50 - i*0.2 for i in range(264)]
}

# Select indicators to plot
selected_indicators = ['GDP per capita', 'Health', 'Population']

# Create scatter plot for each indicator against population
for indicator in selected_indicators[:-1]:
    if indicator in median_values:
        # Downsample data by selecting every 10th data point
        x = population_values[::10]
        y = median_values[indicator][:264][::10]
        plt.scatter(x, y, label=indicator)

plt.legend(labels=selected_indicators[:-1])

# Add labels and legend
plt.xlabel('Population')
plt.ylabel('Median Value')
plt.title('Median Values of Selected Indicators Against Population')
plt.legend()

# Show plot
plt.show()

# Create dictionary with mode values
mode_values = {
    'Arable land (% of land area)': 11.98,
    'Electricity consumption (kWh per capita)': 99.3,
    'Percentage of GDP': 23.9,
    'Population': 38928346,
    'Life expectancy at birth (years)': 76.1,
    'CO2 emissions (metric tons per capita)': 4.15,
    'Internet users (per 100 people)': 14.5,
    'Mobile cellular subscriptions (per 100 people)': 62.7
}

# Convert dictionary to pandas DataFrame
df = pd.DataFrame.from_dict(mode_values, orient='index', columns=['Mode'])

# Create heatmap using seaborn
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
ax = sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='.2f')

# Add labels and title
plt.xlabel('')
plt.ylabel('')
plt.title('Mode Values of Selected Indicators', fontsize=16)

# Show plot
plt.show()

#For Creating BarStacked Plot
years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
arable_land_area = [30000, 32000, 35000, 38000, 40000, 42000, 44000, 47000, 50000]
forest_land_area = [12000, 15000, 18000, 21000, 24000, 27000, 30000, 34000, 38000]

# Create figure and axis objects
fig, ax = plt.subplots()

# Set the x-axis ticks and the width of the bars
x_pos = np.arange(len(years))
bar_width = 0.8

# Create the stacked bars
ax.bar(x_pos, arable_land_area, color='green', label='Arable land')
ax.bar(x_pos, forest_land_area, bottom=arable_land_area, color='brown', label='Forest land')

# Add title and labels to axis object
ax.set_title('Land Area Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Land Area (sq. km)')

# Set the x-axis ticks and labels
ax.set_xticks(x_pos)
ax.set_xticklabels(years)
ax.tick_params(axis='x', rotation=45)

# Add legend to axis object
ax.legend()

# Show plot
plt.show()








