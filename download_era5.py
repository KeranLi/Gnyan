import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "fraction_of_cloud_cover",
        "relative_humidity",
        "specific_humidity",
        "specific_rain_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": [
        "1980", "1981", "1982",
        "1983", "1984", "1985",
        "1986", "1987", "1988",
        "1989"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "07", "14", "21",
        "28"
    ],
    "time": ["14:00"],
    "pressure_level": ["500", "1000"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [37, 78, 26, 102]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
