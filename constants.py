NUMERICAL_ATTRIBUTES = \
    ["x",
     "y",
     "age_category",
     "speed_limit"]

CATEGORICAL_COLUMNS = \
    ['is_drunk_biker',
     'biker_location',
     'gender',
     'intersection_type',
     'month',
     'is_drunk_driver',
     'vehicle_type',
     'light_condition',
     'locality',
     'road_surface_type',
     'weather']

PROPENSITY_MODEL_FEATURES = \
    ['month',
    'speed_limit', 
    'biker_location_Bike Lane / Paved Shoulder',
    'biker_location_Non-Roadway',
    'biker_location_Sidewalk / Crosswalk / Driveway Crossing',
    'biker_location_Travel Lane', 'intersection_type_Intersection',
    'intersection_type_Intersection-Related',
    'intersection_type_Non-Intersection', 'intersection_type_Non-Roadway',
    'light_condition_Dark - Lighted Roadway',
    'light_condition_Dark - Roadway Not Lighted',
    'light_condition_Daylight', 'locality_Mixed (30% To 70% Developed)',
    'locality_Rural (<30% Developed)', 'locality_Urban (>70% Developed)',
    'weather_Clear', 'weather_Cloudy', 'weather_Rain']


T_CATEGORY = 'road_condition'
T_BINARY = 'is_wet'
Y = 'severity'