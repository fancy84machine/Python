
# coding: utf-8

# In[1]:


#Clean File
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_name):
    
    # read the data file
    df = pd.read_csv(file_name, index_col=False, header=None)
    
    # define the column labels for the real data file from NOAA: column_labels
    column_labels = "Wban,date,Time,StationType,sky_condition,sky_conditionFlag,visibility,visibilityFlag,wx_and_obst_to_vision,wx_and_obst_to_visionFlag,dry_bulb_faren,dry_bulb_farenFlag,dry_bulb_cel,dry_bulb_celFlag,wet_bulb_faren,wet_bulb_farenFlag,wet_bulb_cel,wet_bulb_celFlag,dew_point_faren,dew_point_farenFlag,dew_point_cel,dew_point_celFlag,relative_humidity,relative_humidityFlag,wind_speed,wind_speedFlag,wind_direction,wind_directionFlag,value_for_wind_character,value_for_wind_characterFlag,station_pressure,station_pressureFlag,pressure_tendency,pressure_tendencyFlag,presschange,presschangeFlag,sea_level_pressure,sea_level_pressureFlag,record_type,hourly_precip,hourly_precipFlag,altimeter,altimeterFlag"
    
    # define the sub-set list of columns to drop
    list_to_drop = ['sky_conditionFlag', 'visibilityFlag', 'wx_and_obst_to_vision', 'wx_and_obst_to_visionFlag',
                    'dry_bulb_farenFlag', 'dry_bulb_celFlag', 'wet_bulb_farenFlag', 'wet_bulb_celFlag', 'dew_point_farenFlag',
                    'dew_point_celFlag', 'relative_humidityFlag', 'wind_speedFlag', 'wind_directionFlag', 'value_for_wind_character',
                    'value_for_wind_characterFlag', 'station_pressureFlag', 'pressure_tendencyFlag', 'pressure_tendency',
                    'presschange','presschangeFlag', 'sea_level_pressureFlag', 'hourly_precip',  'hourly_precipFlag', 'altimeter', 'record_type', 'altimeterFlag', 'junk']
    
    # split on the comma to create a list, then append one more element as the data files all have a trailing comma at the end of each line: column_labels_list
    column_labels_list = column_labels.split(",")
    column_labels_list.append("junk")
                              
    # assign the new colum labels to the dataframe
    df.columns = column_labels_list
                              
    # Drop all the column data that we don't need
    df = df.drop(list_to_drop, axis='columns')
                              
    # Clean the `Time` column to prepare for creating a date-time index:
    # Pad with leading (left) zeros, for hours
    df['Time'] = df['Time'].apply(lambda x: '{:0>4}'.format(x))
                              
    # Clean the `Time` column to prepare for creating a date-time index:
    # Pad with trailing (right) zeros, for seconds
    df['Time'] = df['Time'].apply(lambda x: '{:<06}'.format(x))
                              
    # Convert the `date` column to a string so it can be combined with the `Time` column
    df['date'] = df['date'].apply(str)
                              
    # Create a new date-time container from the updated `date` and `Time` columns: date_times
    date_times = pd.to_datetime( df['date'] + " " + df['Time'], format="%Y%m%d %H%M%S" )
                                
    # Set the DataFrame index to this new `date_times` container:
    df.set_index(date_times, inplace=True)
                                
    for col in ['wind_speed','dry_bulb_faren','dew_point_faren']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
                                
file_name = "https://s3.amazonaws.com/assets.datacamp.com/production/course_1639/datasets/NOAA_QCLCD_2011_hourly_13904.txt"
df_clean = read_data(file_name)
                                
climatology_data_file = "https://s3.amazonaws.com/assets.datacamp.com/production/course_1639/datasets/weather_data_austin_2010.csv"
df_climate = pd.read_csv(climatology_data_file, parse_dates=True, index_col='Date')
    


# In[2]:


#Signal variance
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())


# In[3]:


#Sunny or cloudy
# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean['sky_condition']=='CLR']

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())


# In[4]:


#Weekly average temperature and visibility
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()
                                      
# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()


# In[5]:


#Daily hours of clear sky
# Create a Boolean Series for sunny days: sunny
sunny = df_clean['sky_condition'] == 'CLR'

# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample('D').sum()

# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample('D').count()

# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()


# In[6]:


#Heat or humidity
# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)

# Show the plot
plt.show()


# In[7]:


##Probability of high temperatures
# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-Aug','Temperature'].max()
print(august_max)

# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011
august_2011 = df_clean.loc['2011-Aug','dry_bulb_faren'].resample('D').max()

# Filter out days in august_2011 where the value exceeded august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011 > august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)

# Display the plot
plt.show()

