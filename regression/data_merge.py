import numpy as np
import csv
import sys
from dateutil.parser import parse
from dateutil.parser import parserinfo
from datetime import timedelta

def read_data(filepath):
    rows = []
    with open(filepath, 'rU') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONE, dialect=csv.excel_tab)
        for row in reader:
            rows.append(row[0].split(','))
    return rows

def merge(a, b):
    return a + b

def reformat_date(date):
    parts = date.replace("\"", "").split('/')
    new_date =  parts[1] + "/" + parts[0] + "/" + parts[2]
    return new_date

def eat_to_utc(ea_time):
    return ea_time + timedelta(hours=-3)

if __name__ == "__main__":

    alliance_filepath = "ML_project/kibera_sub.csv"
    print ("Fetching Air Data...")
    alliance_data = read_data(alliance_filepath)
    alliance_headers = alliance_data[0][3:]
    alliance_points = alliance_data[1:]
    alliance_map = {}
    print ("Reading Air Time Stamps...")
    for point in alliance_points:
        try:
            date = point[1].replace("\"", "")
            time = point[2].replace("\"", "")
            timestamp = parse(date + " " + time, dayfirst=True).replace(second=0, microsecond=0)
            converted_ts = eat_to_utc(timestamp)
            alliance_map[converted_ts] = point[2:]
        except ValueError:
            continue

    wind_filepath = "Wind.csv"
    print ("Fetching Wind Data...")
    wind_data = read_data(wind_filepath)
    wind_headers = wind_data[0]
    wind_points = wind_data[1:]
    mapped_wind_points = []
    print ("Reading Wind Time Stamps...")
    for point in wind_points:
        try:
            entry = point[0].replace("\"", "")
            entry = entry.split(" ")
            date = reformat_date(entry[0])
            time = entry[1]
            timestamp = parse(date + " " + time, dayfirst=True).replace(second=0, microsecond=0)
            mapped_wind_points.append([timestamp] + point[1:])
        except ValueError:
            continue

    num_points = len(mapped_wind_points)
    errors = 0

    merged = []
    new_headers = wind_headers + alliance_headers

    print ("Merging Data Points...")

    for point in mapped_wind_points:
        try:
            m = merge(point, alliance_map[point[0]][1:])
            merged.append(m)
        except KeyError:
            errors += 1

    print (len(mapped_wind_points))
    print (errors)

    out_filepath = "Kibera_Merged.csv"
    with open(out_filepath, "wb") as f:
        writer = csv.writer(f)
        writer.writerows([new_headers] + merged)



