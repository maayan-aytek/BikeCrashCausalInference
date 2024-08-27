import pandas as pd
import warnings
warnings.simplefilter("ignore")


def data_processing(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    cols_to_remove = ['OBJECTID', 'X', 'Y', 'BikeAge', 'AmbulanceR', 'BikeAlcDrg', 'BikeDir', 'BikeRace', 'CrashAlcoh', 'CrashDay', 'CrashGrp', 'CrashHour', 'BikeInjury', 'CrashType', 'CrashYear', 'Developmen',
                'DrvrAge', 'DrvrAgeGrp', 'DrvrAlcDrg','DrvrInjury', 'DrvrRace', 'DrvrSex','HitRun', 'NumBicsAin', 'NumBicsBin', 'NumBicsCin', 'NumBicsKil', 'NumBicsNoi',
                'NumBicsTot', 'NumBicsUin', 'NumLanes', 'NumUnits', 'RdCharacte', 'RdClass', 'RdConfig', 'RdDefects', 'RdFeature', 'Region', 'TraffCntrl', 'Workzone', 'City', 'County', 'RuralUrban']
    
    filter_values = {'BikeAgeGrp': ['Unknown'], 'BikeAlcFlg': ['Missing', 'Unknown'],
                'BikePos': ['Unknown', 'Other'], 'CrashLoc': ['Unknown Location'],
                'CrashSevr': ['Unknown Injury'],
                'DrvrVehTyp': ['Unknown'],
                'DrvrAlcFlg': ['Missing', 'Unknown'],
                'LightCond': ['Unknown', 'Other', 'Dark - Unknown Lighting'], 'RdConditio': ['Unknown', 'Water (Standing, Moving)', 'Sand, Mud, Dirt, Gravel', 'Snow', 'Ice', 'Other'],
                'RdSurface': ['Missing', 'Other', 'Unknown', 'Gravel', 'Soil', 'Sand'],
                'SpeedLimit': ['Unknown'],
                'Weather': ['Other'],
                'BikeSex': ['Unknown']}
    
    replace_values = {
                'RdSurface': {'Grooved Concrete': 'Concrete'},
                'BikePos': {'Multi-use Path': 'Bike Lane / Paved Shoulder',
                            'Driveway / Alley': 'Non-Roadway'},
                'DrvrVehTyp': {
                    "Sport Utility": "Passenger Vehicle",
                    "Light Truck (Mini-Van, Panel)": "Light Truck",
                    "Pickup": "Light Truck",
                    "Passenger Car": "Passenger Vehicle",
                    "Van": "Light Truck",
                    "Taxicab": "Passenger Vehicle",
                    "Motorcycle": "Two-Wheeler",
                    "Moped": "Two-Wheeler",
                    "Tractor/Semi-Trailer": "Heavy Truck",
                    "Truck/Trailer": "Heavy Truck",
                    "Single Unit Truck (2-Axle, 6-Tire)": "Heavy Truck",
                    "Single Unit Truck (3 Or More Axles)": "Heavy Truck",
                    "Unknown Heavy Truck": "Heavy Truck",
                    "Truck/Tractor": "Heavy Truck",
                    "Motor Home/Recreational Vehicle": "Heavy Truck",
                    "EMS Vehicle, Ambulance, Rescue Squad": "Light Truck",
                    "Firetruck": "Heavy Truck",
                    "Police": "Passenger Vehicle",
                    "School Bus": "Heavy Truck",
                    "Other Bus": "Heavy Truck",
                    "Commercial Bus": "Heavy Truck",
                    "Activity Bus": "Heavy Truck"},
                'LightCond': {'Dusk': 'Daylight', 'Dawn': 'Daylight'},
                'Weather': {'Fog, Smog, Smoke': 'Cloudy', 'Snow, Sleet, Hail, Freezing Rain/Drizzle': 'Rain'}
                }
    
    rename_cols_dict = {'BikeAgeGrp': 'age_category', 'BikeAlcFlg': 'is_drunk_biker', 'BikePos': 'biker_location', 'BikeSex': 'gender', 'CrashLoc': 'intersection_type', 'CrashMonth': 'month', 'CrashSevr': 'severity',
                    'DrvrAlcFlg': 'is_drunk_driver', 'DrvrVehTyp': 'vehicle_type', 'Latitude': 'y',  'LightCond': 'light_condition', 'Locality': 'locality', 'Longitude': 'x',
                    'RdConditio': 'road_condition', 'RdSurface': 'road_surface_type', 'SpeedLimit': 'speed_limit', 'Weather': 'weather'}

    df = df.drop(cols_to_remove, axis=1)
    for column, values in filter_values.items():
        df = df[~df[column].isin(values)]

    for col, replace_dict in replace_values.items():
        df[col] = df[col].replace(replace_dict)
    df = df.rename(columns=rename_cols_dict)
    return df


def features_engineer(data_path: str) -> pd.DataFrame:
    df = data_processing(data_path).copy()
    df['is_wet'] = df['road_condition'].apply(lambda x: 1 if x == 'Wet' else 0)
    df['is_drunk_biker'] = df['is_drunk_biker'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['is_drunk_driver'] = df['is_drunk_driver'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['is_male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['age_category'] = df['age_category'].apply(lambda x: int(x.split("-")[0]) if "-" in x else int(x.split("+")[0]))
    df['speed_limit'] = df['speed_limit'].apply(lambda x: int(x.split(" ")[0]))
    severity_dict = {"O: No Injury": 0, "C: Possible Injury": 1, "B: Suspected Minor Injury": 2, "A: Suspected Serious Injury": 3, "K: Killed": 4}
    df['severity'] = df['severity'].replace(severity_dict)
    month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month'] = df['month'].replace(month_mapping)
    df = df.drop(['road_condition', 'gender'], axis=1)
    category_cols = ['biker_location', 'intersection_type', 'vehicle_type', 'light_condition', 'locality', 'road_surface_type', 'weather']
    df = pd.get_dummies(df, columns=category_cols)
    return df 