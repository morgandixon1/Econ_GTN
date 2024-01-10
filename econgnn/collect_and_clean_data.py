import requests
import json
import csv
import os
import requests
import json
import pandas as pd
import geopandas as gpd
import random
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
import io
from Levenshtein import distance
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet  # You'll need to install the nltk package and download 'wordnet' if you haven't already
from fuzzywuzzy import process
import openai
import glob
import pgeocode
from geopy.geocoders import Nominatim
import numpy as np
import time  # Import the time module at the top of your script if not already done

openai.api_key = "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"

class GNNProcessor:
    def __init__(self, api_key, radius, business_type, limit, filename):
        self.api_key = api_key
        self.radius = radius
        self.business_type = business_type
        self.limit = limit  # Make sure this is the maximum allowed by the API per call
        self.filename = filename
        self.total_count = 0
        self.load_existing_data()
        print(f"Pandas version: {pd.__version__}")  # Debugging Line 1: Print Pandas version

    def load_existing_data(self):
        print("Trying to read from: ", os.path.abspath("/Users/morgandixon/Desktop/businesses.csv"))  # Debugging line
        fieldnames = ['Name', 'Website', 'Search Keywords', 'Street Address', 'Latitude', 'Longitude', 'Number of Reviews', 'Node Neighborhoods', 'EGRP']
        try:
            self.existing_data = pd.read_csv("/Users/morgandixon/Desktop/businesses.csv")  # Hardcoding the file path
        except pd.errors.ParserError:
            print("Error reading existing CSV. Creating a new one.")
            self.existing_data = pd.DataFrame(columns=fieldnames)
    def is_duplicate(self, business):
        if self.existing_data.empty:
            return False
        return any((self.existing_data['Name'] == business['name']) & 
                   (self.existing_data['Latitude'] == business['geometry']['location']['lat']) & 
                   (self.existing_data['Longitude'] == business['geometry']['location']['lng']))
        
    def fetch_businesses(self, location, radius):
        fetched_businesses = []
        endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "key": self.api_key,
            "location": location,
            "radius": radius,
            "type": ""  # Removing the business_type to get all kinds of businesses
        }
        res = requests.get(endpoint, params=params)
        return json.loads(res.text), fetched_businesses

    def process_business_details(self, businesses):
        for business in businesses.get('results', []):
            place_id = business.get('place_id')
            if place_id:
                details_endpoint = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=website,user_ratings_total,formatted_address,types&key={self.api_key}"
                details_res = requests.get(details_endpoint)
                details = json.loads(details_res.text)
                website = details.get('result', {}).get('website', '')
                user_ratings_total = details.get('result', {}).get('user_ratings_total', 0)
                street_address = details.get('result', {}).get('formatted_address', '')
                types = details.get('result', {}).get('types', [])

                business['Website'] = website
                business['Number of Reviews'] = user_ratings_total
                business['Street Address'] = street_address
                business['Types'] = types  # New

    def save_to_csv(self, bounding_boxes, all_businesses):
        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = self.existing_data.columns.tolist()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.stat(self.filename).st_size == 0:
                writer.writeheader()
            for bounding_box in bounding_boxes:
                if self.total_count >= self.limit:
                    break
                for place in all_businesses:
                    name = place.get('name', 'Unknown')
                    if name in self.existing_data['Name'].values:
                        continue

                    latitude = place.get('geometry', {}).get('location', {}).get('lat', None)
                    longitude = place.get('geometry', {}).get('location', {}).get('lng', None)
                    website = place.get('Website', '')
                    number_of_reviews = place.get('Number of Reviews', 0)
                    street_address = place.get('Street Address', '')
                    
                    # Here we add 'Search Keywords'
                    row = {
                        'Name': name,
                        'Website': place.get('Website', ''),  # Modified
                        'Search Keywords': ', '.join(place.get('Types', [])),  # Modified
                        'Street Address': street_address,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Number of Reviews': number_of_reviews,
                        'Node Neighborhoods': '',
                        'EGRP': ''
                    }
                    print(f"Adding Business: {row}")

                    writer.writerow(row)
                    self.total_count += 1

                    if isinstance(self.existing_data, pd.DataFrame):
                        print(type(self.existing_data))  # Debugging line
                        self.existing_data = self.existing_data._append(row, ignore_index=True)
                    else:
                        print("Error while saving to CSV: 'existing_data' is not a DataFrame")

    def generate_random_coordinates(self, min_lat, max_lat, min_lng, max_lng):
        latitude = random.uniform(min_lat, max_lat)
        longitude = random.uniform(min_lng, max_lng)
        return f"{latitude},{longitude}"

    def add_new_businesses(self, bounding_boxes):
        new_count = 0  # Track how many new businesses are added
        all_businesses = []
        print("Entering add_new_businesses function")  # Debugging line
        
        for min_lat, max_lat, min_lng, max_lng in bounding_boxes:
            if new_count >= 500:  # Exit if we have enough businesses
                break

            # Generate one random coordinate set per bounding box
            random_coordinates = self.generate_random_coordinates(min_lat, max_lat, min_lng, max_lng)
            print(f"Fetching businesses for location: {random_coordinates}")  # Debugging line

            # Increase radius to 5 miles ~ 8046 meters
            fetched_data, fetched_businesses = self.fetch_businesses(random_coordinates, 8046)

            for business in fetched_data.get('results', []):
                if not self.is_duplicate(business):
                    all_businesses.append(business)
                    new_count += 1
                    if new_count >= 500:  # Break if we have enough businesses
                        break

        print("Exiting loop, moving to process_business_details")  # Debugging line
        self.process_business_details({'results': all_businesses})
        self.save_to_csv(bounding_boxes, all_businesses)

class AssociateEGRP:
    def __init__(self, businesses_filename, industry_filename):
        self.businesses_filename = businesses_filename
        self.industry_filename = industry_filename
        self.load_business_data()
        self.load_industry_data()

    def load_business_data(self):
        try:
            self.business_data = pd.read_csv(self.businesses_filename)
            #print("Sample of Business Data:")
            #print(self.business_data.head())
        except Exception as e:
            print(f"Error reading existing Business CSV: {e}")
            self.business_data = pd.DataFrame()

    def load_industry_data(self):
        try:
            self.industry_data = pd.read_csv(self.industry_filename)
            #print("Sample of Industry Data:")
            #print(self.industry_data.head())
        except Exception as e:
            print(f"Error reading existing Industry CSV: {e}")
            self.industry_data = pd.DataFrame()

    def generate_ngrams(self, text, n=2):
        if text and len(text.split()) >= n:  # Make sure the text has enough words for n-grams
            vectorizer = CountVectorizer(ngram_range=(n, n))
            ngram_matrix = vectorizer.fit_transform([text])
            return list(vectorizer.get_feature_names_out())
        else:
            return []  # Return an empty list if text is None or not long enough for n-grams

    def associate_egrp_data(self):
        try:
            self.business_data = pd.read_csv(self.businesses_filename)
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return

        if self.business_data.empty or self.industry_data.empty:
            print("Data is not loaded properly.")
            return

        manual_entries = set()
        next_business_to_process = 0

        try:
            for i in range(1, 7001):
                debug_businesses = self.business_data[(self.business_data['NAICS'].isna()) | (self.business_data['NAICS'] == -1)]
                business_to_update = debug_businesses.iloc[next_business_to_process:next_business_to_process+1]

                if business_to_update.empty:
                    print("All businesses have been processed.")
                    break

                index_to_update = business_to_update.index[0]
                business_name = business_to_update['Name'].iloc[0]
                business_address = business_to_update['Street Address'].iloc[0]

                print(f"Processing business {i}/7000: {business_name}")
                time.sleep(1)  # Pause for 1 second

                next_business_to_process += 1
                business_website = business_to_update['Website'].iloc[0]
                business_keywords = business_to_update['Search Keywords'].iloc[0]
                query_content = f"Generate a likely NAICS code for the given information:\"{business_name}, {business_address}, {business_website}, {business_keywords}\""

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "assistant", "content": query_content}],
                    temperature=0,
                    max_tokens=60,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                content = response['choices'][0]['message']['content']
                chosen_naics = ''.join(filter(str.isdigit, content))

                if chosen_naics == '':
                    print(f"NAICS code not found for {business_name}. Will handle manually.")
                    self.business_data.at[index_to_update, 'NAICS'] = -1
                    manual_entries.add(business_name)
                else:
                    matching_industry_row = self.industry_data[self.industry_data['NAICS'] == int(chosen_naics)]
                    if not matching_industry_row.empty:
                        chosen_industry = matching_industry_row['Description'].iloc[0]
                        self.business_data.at[index_to_update, 'NAICS'] = int(chosen_naics)
                        self.business_data.at[index_to_update, 'Industry'] = chosen_industry
                        print(f"Successfully received NAICS code {chosen_naics} for {business_name}.")
                    else:
                        print(f"NAICS code not found in industry table for {business_name}. Will handle manually.")
                        self.business_data.at[index_to_update, 'NAICS'] = -1
                        manual_entries.add(business_name)

                self.business_data.to_csv(self.businesses_filename, index=False)

        except Exception as e:
            print(f"An error occurred: {e}")

        # Handle the manual entries
        for business_name in manual_entries:
            print(f"Manually enter NAICS code for {business_name}:")
            while True:
                chosen_naics = input("Enter correct NAICS code: ")
                matching_industry_row = self.industry_data[self.industry_data['NAICS'] == int(chosen_naics)]
                if not matching_industry_row.empty:
                    chosen_industry = matching_industry_row['Description'].iloc[0]
                    break
                else:
                    print("Invalid NAICS code. Please try again.")

            self.business_data.loc[self.business_data['Name'] == business_name, 'NAICS'] = int(chosen_naics)
            self.business_data.loc[self.business_data['Name'] == business_name, 'Industry'] = chosen_industry
            self.business_data.to_csv(self.businesses_filename, index=False)

class DataDistribution:
    def __init__(self):
        print("Initializing DataDistribution...")
        self.geojson_loaded = False  # Add this flag to check if GeoJSON is loaded

    def load_geojson(self):
        if not self.geojson_loaded:  # Check the flag here
            try:
                print("Attempting to load GeoJSON file...")
                self.gdf = gpd.read_file("/Users/morgandixon/Desktop/zipcodeboundries.geojson")
                print("Successfully loaded GeoJSON file.")
                self.geojson_loaded = True  # Update the flag when successfully loaded
            except Exception as e:
                print(f"Failed to load GeoJSON file. Error: {e}")

    def get_zip_from_lat_lon_geospatial(self, lat, lon):
        point = gpd.GeoDataFrame({'geometry': [gpd.points_from_xy([lon], [lat])[0]]},
                                index=[0], crs=self.gdf.crs)
        location_data = gpd.sjoin(point, self.gdf, op='within')
        
        if not location_data.empty:
            return location_data['ZIP_CODE'].values[0]
        else:
            return "Zip code not found"

    def dfforeachzip(self, file_path):
        self.business_dfs = {}
        print("Checking for missing Zip codes...")
        
        df = pd.read_csv(file_path)
        
        if df.empty:
            print("The DataFrame is empty. Exiting function.")
            return self.business_dfs
        
        if 'Zip Code' in df.columns:
            df = df[df['Zip Code'] != 'Zip code not found']

        missing_zips = df['Zip Code'].isna().sum()
        if missing_zips > 0:
            print("Missing Zip codes found. Loading GeoJSON for geospatial lookup.")
            self.load_geojson()
            def get_zip(row):
                lat = row['Latitude']
                lon = row['Longitude']
                zip_code = self.get_zip_from_lat_lon_geospatial(lat, lon)
                return zip_code

            new_zip_codes = df.apply(get_zip, axis=1)
            df['Zip Code'] = new_zip_codes
            df.to_csv(file_path, index=False)
        else:
            print("All businesses have Zip codes. Skipping geospatial lookup.")
        
        zip_code_groups = df.groupby('Zip Code')
        for name, group in zip_code_groups:
            self.business_dfs[name] = group
        
        return self.business_dfs  # Comment this line if it's causing the DataFrame to display

    def dfforindustrytable(self):
        industry_dfs = {}
        directory_path = "zip_code_industry_tables"  # Adjust the path as per your setup
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return

        industry_zip_dfs = {}  # Initialize an empty dictionary to store industry DataFrames by ZIP code
        
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                zip_code = filename.split('.')[0]
                try:
                    df = pd.read_csv(f"{directory_path}/{filename}")
                except Exception as e:
                    print(f"Could not read industry CSV for Zip code {zip_code}. Error: {e}")
                    continue
                
                #print(f"Data for Zip code {zip_code}:")
                #print(df.head())  # Print first 5 rows to check
                industry_dfs[zip_code] = df
        
        return industry_dfs  # Return the dictionary of ZIP code industry DataFrames

    def accuracyadjustments(self, file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Could not read CSV file. Error: {e}")
            return
        num_businesses_in_df = len(df)
        total_known_businesses = 24264  
        adjustment_factor = num_businesses_in_df / total_known_businesses
        print(f"Number of businesses in DataFrame: {num_businesses_in_df}")
        print(f"Total known businesses in Spokane + Kootenai County: {total_known_businesses}")
        print(f"Adjustment factor: {adjustment_factor}")
        return adjustment_factor

    def egrpdistribution(self, business_dfs, industry_dfs, adjustment_factor, file_path):
        print("Executing EGRP distribution function...")
        if business_dfs is None or industry_dfs is None:
            print("Error: One of the data frames is None. Exiting function.")
            return
        concatenated_dfs = []
        for zip_code, business_df in business_dfs.items():  # Loop through all zip codes
            str_zip_code = str(zip_code)  # Convert ZIP code to string
            if str_zip_code == 'Zip code not found':
                continue
            print(f"Processing ZIP code: {str_zip_code}")
            industry_df = industry_dfs.get(str_zip_code, None)  # Use string ZIP code here
            if industry_df is None:
                print(f"No industry data available for ZIP code {str_zip_code}. Exiting.")
                continue
            if '2022 GRP Earnings' not in industry_df.columns:
                print(f"'2022 GRP Earnings' column not found in industry data for ZIP code {zip_code}. Exiting.")
                return
            if 'Number of Reviews' not in business_df.columns:
                print(f"'Reviews' column not found in business data for ZIP code {zip_code}. Exiting.")
                return

            print(f"Found industry DF and Business DF for ZIP code {zip_code}")
            business_naics = set(business_df['NAICS'].astype(int))
            industry_naics = set(industry_df['NAICS'])
            common_naics = business_naics.intersection(industry_naics)

            if not common_naics:
                print(f"No common NAICS codes found for ZIP code {zip_code}. Exiting.")
                return

            for naics_int in common_naics:
                egrp_zip_naics = industry_df.loc[industry_df['NAICS'] == naics_int, '2022 GRP Earnings'].values[0]
                adj_egrp_zip_naics = egrp_zip_naics * adjustment_factor

                business_subgroup = business_df[business_df['NAICS'].astype(int) == naics_int]
                num_businesses = len(business_subgroup)
                total_reviews = business_subgroup['Number of Reviews'].sum()

                if num_businesses == 1 and (pd.isna(total_reviews) or total_reviews == 0):
                    business_name = business_subgroup.iloc[0]['Name']
                    print(f"{business_name} makes up 100% of the {naics_int} industry with 0/0 reviews. The industry GRP is {egrp_zip_naics}, the distributed and adjusted EGRP for this business is {adj_egrp_zip_naics}")
                    continue

                if pd.isna(total_reviews) or total_reviews == 0:
                    continue

                egrp_dict = {}
                for idx, business in business_subgroup.iterrows():
                    business_name = business['Name']
                    reviews_this_business = business['Number of Reviews']
                    if pd.isna(reviews_this_business):
                        continue

                    market_share_this_business = (reviews_this_business / total_reviews) * 100
                    egrp_this_business = (reviews_this_business / total_reviews) * adj_egrp_zip_naics
                    print(f"ZIP code {zip_code}: {business_name} makes up {market_share_this_business:.2f}% of the {naics_int} industry with {reviews_this_business}/{total_reviews} reviews. The industry GRP is {egrp_zip_naics}, the distributed and adjusted EGRP for this business is {egrp_this_business}")
                    egrp_dict[idx] = egrp_this_business

                if egrp_dict:
                    business_df.loc[list(egrp_dict.keys()), 'Estimated EGRP'] = pd.Series(egrp_dict)

            concatenated_dfs.append(business_df)
            
        if concatenated_dfs:
            pd.concat(concatenated_dfs).to_csv(file_path, index=False)


if __name__ == "__main__":
    user_choice = input("Select 1890 to add more businesses (very expensive), select 2 to associate EGRP data, select 3 for data distribution: ")
    
    if user_choice == '1890':
        # min_latitude, max_latitude, min_longitude, max_longitude
        spokane_bounding_box = [47.5886, 47.7312, -117.5314, -117.2822]
        kootenai_bounding_box = [47.5881, 47.7732, -116.9211, -116.5483]

        bounding_boxes = [spokane_bounding_box, kootenai_bounding_box]

        gnn_processor = GNNProcessor(
            api_key="NOPE_OUT Of MOney",
            radius=1000,
            business_type='',  # Set to empty string to fetch all types of businesses
            limit=2500,  # You can increase this limit
            filename="/Users/morgandixon/Desktop/businesses.csv"
        )
        prompt_interval = 500  # Number of businesses to fetch before prompting the user
        next_prompt = prompt_interval  # Initialize next prompt point

        while gnn_processor.total_count < gnn_processor.limit:
            previous_count = gnn_processor.total_count  # Store the count before adding new businesses
            
            gnn_processor.add_new_businesses(bounding_boxes)
            
            new_count = gnn_processor.total_count - previous_count  # Calculate how many new businesses were added
            print(f"New businesses added this round: {new_count}")
            print(f"Total businesses added: {gnn_processor.total_count}")

            # Prompt the user when reaching or exceeding 'next_prompt'
            if gnn_processor.total_count >= next_prompt:
                continue_loop = input(f"Do you want to continue fetching businesses? (y/n): ")
                if continue_loop.lower() != 'y':
                    break
                next_prompt += prompt_interval  # Update the next prompt point

        print(f"Final total businesses added: {gnn_processor.total_count}")

    elif user_choice == '2':
        associate_egrp = AssociateEGRP(
            "/Users/morgandixon/Desktop/businesses3.csv", 
            "/Users/morgandixon/Downloads/industrytable4.csv"
        )
        associate_egrp.associate_egrp_data()

    elif user_choice == '3':
        business_file_path = "/Users/morgandixon/Desktop/businesses3.csv"
        dd = DataDistribution()
        # dd.load_geojson()  # Comment out or remove this line
        business_dfs = dd.dfforeachzip("/Users/morgandixon/Desktop/businesses3.csv")
        #print("Loaded business data for ZIP codes:", business_dfs.keys())
        industry_dfs = dd.dfforindustrytable()
        #print("Loaded industry data for ZIP codes:", industry_dfs.keys())
        adjustment_factor = dd.accuracyadjustments("/Users/morgandixon/Desktop/businesses3.csv")
        dd.egrpdistribution(business_dfs, industry_dfs, adjustment_factor, business_file_path)

    else:
        print("Invalid choice. Exiting.")