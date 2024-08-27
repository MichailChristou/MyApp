import pandas as pd

# Load the dataset
file_path_r1 = r'E:\Gorillino\Unipi\_Πτυχιακη\Dataset\DatasetOneFile\MyAppDS.csv'
data = pd.read_csv(file_path_r1, nrows=100)

print(data.columns)

print(data.head(1))


#columns_to_drop = ['POTENTIAL_ENERGY_RATING', 'POTENTIAL_ENERGY_EFFICIENCY', 'ENVIRONMENT_IMPACT_POTENTIAL', 'ENERGY_CONSUMPTION_POTENTIAL','CO2_EMISS_CURR_PER_FLOOR_AREA', 'CO2_EMISSIONS_POTENTIAL', 'LIGHTING_COST_POTENTIAL', 'HEATING_COST_POTENTIAL','HOT_WATER_COST_POTENTIAL', 'HOT_WATER_COST_CURRENT', 'WINDOWS_ENV_EFF', 'WALLS_ENERGY_EFF','WALLS_ENV_EFF', 'ROOF_ENERGY_EFF', 'ROOF_ENV_EFF', 'MAINHEATC_ENERGY_EFF','MAINHEATC_ENV_EFF', 'LIGHTING_ENERGY_EFF', 'LIGHTING_ENV_EFF', 'IMPROVEMENT_ITEM2', 'IMPROVEMENT_ITEM3', 'IMPROVEMENT_ID1', 'IMPROVEMENT_ID2', 'IMPROVEMENT_ID3']  # Replace 'column1', 'column2', etc. with the actual column names you want to drop
#data.drop(columns=columns_to_drop, inplace=True)


