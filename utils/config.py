# project/src/config.py
"""
Configuration file to store data paths.
Paths are sourced from the project specification document, Table 1.
"""

# S3 bucket URI 
S3_BUCKET_PATH = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data"

DATA_PATHS = {
    # Los Angeles Crime Data (2010-2019) [cite: 26, 31]
    "crime_data_2010_2019": f"{S3_BUCKET_PATH}/LA_Crime_Data/LA_Crime_Data_2010_2019.csv",
    
    # Los Angeles Crime Data (2020-) [cite: 27, 31]
    "crime_data_2020_present": f"{S3_BUCKET_PATH}/LA_Crime_Data/LA_Crime_Data_2020_2025.csv",
    
    # Census Blocks [cite: 28, 31]
    "census_blocks": f"{S3_BUCKET_PATH}/LA_Census_Blocks_2020.geojson",
    
    # Census Blocks Fields [cite: 29, 31]
    "census_blocks_fields": f"{S3_BUCKET_PATH}/LA_Census_Blocks_2020_fields.csv",
    
    # Median Household Income by Zip Code [cite: 32]
    "median_income": f"{S3_BUCKET_PATH}/LA_Income_2021.csv",
    
    # LA Police Stations [cite: 34]
    "police_stations": f"{S3_BUCKET_PATH}/LA_Police_Stations.csv",
    
    # Race and Ethnicity Codes [cite: 33]
    "race_codes": f"{S3_BUCKET_PATH}/RE_codes.csv",
    
    # MO Codes [cite: 34]
    "mo_codes": f"{S3_BUCKET_PATH}/MO_codes.txt"
}