# Standard library imports
import argparse
import csv
import json
import os
import pickle
import random
import time
from math import cos

# Third-party imports
import eventlet
import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import requests
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, Namespace
from geopy.distance import geodesic
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from numpy import deg2rad
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

matplotlib.use('Agg')
eventlet.monkey_patch()
model_path = "trained_model.h5"
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')
api_key = None
radius = None
business_type = None
limit = None
cut_edges = []

enc = LabelEncoder()

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

class MapRenderer:
    def __init__(self, filename):
        self.filename = filename

    def load_data_into_map(self, m, coords_list=None):
        if coords_list is None:  # If coords_list is not provided, read from file
            coords_list = []
            with open(self.filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        coords = [float(row['Latitude']), float(row['Longitude'])]
                        folium.Marker(coords, popup=row['Name']).add_to(m)
                        coords_list.append((coords, row['Name']))
                    except ValueError:
                        continue
        else:  # If coords_list is provided, use it directly
            for coords, name in coords_list:
                folium.Marker(coords, popup=name).add_to(m)

        return coords_list

class GNNProcessor:
    def __init__(self, model, model_path, api_key, radius, business_type, limit, filename=None):
        self.model = model
        self.api_key = api_key
        self.radius = radius
        self.business_type = business_type
        self.limit = limit
        self.filename = filename if filename else "/Users/morgandixon/Desktop/businesses.csv"
        self.enc = LabelEncoder()
        self.preprocessor = None
        if self.model is None and model_path:  # Using the global model_path
            try:
                if os.path.exists(model_path):  # Directly checking for the model here
                    self.model = load_model(model_path)
                self.preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor
            except OSError:
                print(f"Could not load model from {model_path}. It will be set later.")  # Using the global model_path
                self.model = None

    def fetch_businesses(self, location):
        endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "key": self.api_key,
            "location": location,
            "radius": self.radius,
            "type": self.business_type
        }
        res = requests.get(endpoint, params=params)
        businesses = json.loads(res.text)
        self.process_business_details(businesses)
        return businesses

    def process_business_details(self, businesses):
        for business in businesses.get('results', []):
            place_id = business.get('place_id')
            if place_id:
                details_endpoint = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=website,user_ratings_total,formatted_address&key={self.api_key}"
                details_res = requests.get(details_endpoint)
                details = json.loads(details_res.text)
                website = details.get('result', {}).get('website', '')
                user_ratings_total = details.get('result', {}).get('user_ratings_total', 0)
                street_address = details.get('result', {}).get('formatted_address', '')

                business['Website'] = website
                business['Number of Reviews'] = user_ratings_total
                business['Street Address'] = street_address

    def save_to_csv(self, central_locations):
        existing_data = self.load_existing_data()
        fieldnames = ['Name', 'Website', 'Search Keywords', 'Street Address', 'Latitude', 'Longitude', 'Number of Reviews']
        total_count = 0
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not existing_data:
                writer.writeheader()
            for central_location in central_locations:
                if total_count >= 2500:
                    break
                close_businesses = self.find_close_businesses(central_location)
                for place in close_businesses:
                    if total_count >= 2500:
                        break
                    name = place.get('name', '')
                    latitude = place.get('coords', [None, None])[0]
                    longitude = place.get('coords', [None, None])[1]
                    website = place.get('Website', '')
                    number_of_reviews = place.get('Number of Reviews', 0)
                    street_address = place.get('Street Address', '')

                    row = {
                        'Name': name,
                        'Website': website,
                        'Search Keywords': '',  # Assuming this will be filled in later
                        'Street Address': street_address,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Number of Reviews': number_of_reviews
                    }
                    writer.writerow(row)
                    total_count += 1

    def load_existing_data(self):
        existing_data = set()
        try:
            with open(self.filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_data.add(row.get('Name'))
        except FileNotFoundError:
            pass
        return existing_data

    def find_close_businesses(self, radius_miles=10, num_to_return=20):
        businesses = []
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if is_float(row[4]) and is_float(row[5]):
                    name, lat, lng = row[0], float(row[4]), float(row[5])
                    businesses.append({'name': name, 'coords': (lat, lng)})
                else:
                    print(f"Skipping row due to invalid data: {row}")

        if not businesses:
            print("No valid businesses found.")
            return []
        random.shuffle(businesses)
        central_business = random.choice(businesses)
        close_businesses = [business for business in businesses
                            if geodesic(central_business['coords'], business['coords']).miles <= radius_miles]
        if len(close_businesses) >= num_to_return:
            sampled_businesses = random.sample(close_businesses, num_to_return)
            print(f"Found {num_to_return} businesses within {radius_miles} miles of {central_business['name']}")
            for i, business in enumerate(sampled_businesses, 1):
                print(f"{i}. {business['name']} ({business['coords']})")
            return sampled_businesses
        else:
            print("Not enough close businesses found.")
            return []

    def train_model(self):
        preprocessor = None
        model = None
        try:
            print("Entering train_model()")
            data = pd.read_csv(self.filename)

            # Trim column names
            data.columns = data.columns.str.strip()

            # Define numeric and categorical features
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

            numeric_features = ['Latitude', 'Longitude', 'Number of Reviews']
            categorical_features = ['Name', 'Website', 'Search Keywords', 'Street Address']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            X_train = preprocessor.fit_transform(data)
            X_train = X_train.toarray()  # Convert sparse matrix to dense numpy array
            input_dim = X_train.shape[1]  # Number of features

            print(f"Shape of X_train: {X_train.shape}")  # Debugging line
            Y_train = np.ones(X_train.shape[0])

            # Build model
            model = Sequential([
                Dense(128, activation='relu', input_dim=input_dim),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')  # Output layer with one neuron for binary classification
            ])
            model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

            print("Model training started")
            model.fit(X_train, Y_train, epochs=1, batch_size=32)  # Use Y_train as the target labels
            print("Model training completed")

            # Save the preprocessor and model
            joblib.dump(preprocessor, 'preprocessor.pkl')
            print("Preprocessor saved as 'preprocessor.pkl'.")

            model.save('trained_model.h5')
            print("Model saved as 'trained_model.h5'.")

            self.preprocessor = preprocessor
            self.model = model
            print("Exiting train_model() successfully.")
            return model
        except Exception as e:
            print(f"An error occurred in train_model(): {e}")
            self.preprocessor = preprocessor
            self.model = model
            return model

    def build_and_draw_graph(self, coords_list, threshold=0.5):
        if self.model is None or self.preprocessor is None:
            print("Model or preprocessor is not available. Exiting.")
            return None

        G = nx.Graph()
        for coords1, name1 in coords_list:
            prediction_count = 0
            for coords2, name2 in coords_list:
                if name1 == name2:
                    continue
                if prediction_count >= 10:
                    break

                # Create a DataFrame from the features
                combined_features_df = pd.DataFrame([[
                    "Name1", "Website1", "Keywords1", "Address1", coords1[0], coords1[1], 0  # Using 0 as a placeholder
                ]], columns=['Name', 'Website', 'Search Keywords', 'Street Address', 'Latitude', 'Longitude', 'Number of Reviews'])

                preprocessed_features = self.preprocessor.transform(combined_features_df)
                pred_prob = self.model.predict(preprocessed_features, verbose=0)
                print(f"Prediction probability: {pred_prob}")

                # Assuming you want to check if any value in the pred_prob array is greater than or equal to threshold
                if (pred_prob >= threshold).any():
                    G.add_edge(name1, name2)
                    prediction_count += 1

        return G

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

connected_clients = {}

class MyNamespace(Namespace):
    def on_connect(self):
        connected_clients[request.sid] = self

socketio = SocketIO(app, namespace_class=MyNamespace)

business_features = {}
@app.route('/graph_and_map', methods=['GET'])
def graph_and_map():
    global business_features  # Declare as global to modify it

    print("Entering graph_and_map()")
    m = folium.Map(location=[47.6588, -117.4260], zoom_start=8)
    close_businesses = gnn_processor.find_close_businesses(radius_miles=10)
    for business in close_businesses:
        business_name = business['name']
        business_coords = business['coords']
        business_features[business_name] = {
            'Latitude': business_coords[0],
            'Longitude': business_coords[1],
        }

    sampled_coords_list = [(business['coords'], business['name']) for business in close_businesses]
    G = gnn_processor.build_and_draw_graph(sampled_coords_list)
    if G is None:
        print("Graph could not be built.")
        return jsonify({"error": "Graph could not be built"}), 400  # Return a JSON error response
    edges_data = []
    for edge in G.edges():
        coords1 = next((coords for coords, name in sampled_coords_list if name == edge[0]), None)
        coords2 = next((coords for coords, name in sampled_coords_list if name == edge[1]), None)
        if coords1 and coords2:
            edges_data.append({"node1": edge[0], "node2": edge[1], "coords1": coords1, "coords2": coords2})
            for sid, namespace in connected_clients.items():
                namespace.emit('new_edge', {"coords1": coords1, "coords2": coords2, "node1": edge[0], "node2": edge[1]}, room=sid)
            folium.PolyLine([coords1, coords2], color="blue", weight=2.5, opacity=1).add_to(m)
    graph_data = {
        "nodes": [{"id": node} for node in G.nodes()],
        "links": [{"source": edge[0], "target": edge[1]} for edge in G.edges()]
    }
    print("Exiting graph_and_map()")
    map_html = m._repr_html_()
    return jsonify({
        "map_html": map_html,
        "graph_data": graph_data,
        "mapData": {"coords_list": sampled_coords_list},
        "edgesData": edges_data
    })

G = nx.Graph()
edge_cut_count = 0  # Global variable to count the number of edges cut
cut_edges = []  # list to store cut edges
preprocessor = joblib.load('preprocessor.pkl')
model = load_model('trained_model.h5')

def attempt_and_announce_retrain():
    global model, cut_edges, preprocessor

    csv_file_path = "cut_edges.csv"
    if os.path.exists(csv_file_path):
        cut_edges_df = pd.read_csv(csv_file_path)
    else:
        cut_edges_df = pd.DataFrame(columns=["Node1", "Node2", "Latitude1", "Longitude1", "Latitude2", "Longitude2"])

    x_train = []
    y_train = []

    print("cut_edges list before retraining:", cut_edges)

    for node1, node2, features1, features2 in cut_edges:
        lat1, lon1 = features1['Latitude'], features1['Longitude']
        lat2, lon2 = features2['Latitude'], features2['Longitude']

        new_row = pd.DataFrame({"Node1": [node1], "Node2": [node2],
                                "Latitude1": [lat1], "Longitude1": [lon1],
                                "Latitude2": [lat2], "Longitude2": [lon2]})

        cut_edges_df = pd.concat([cut_edges_df, new_row]).reset_index(drop=True)

        # Prepare data for retraining
        all_columns = ['Latitude', 'Longitude', 'Number of Reviews', 'Name', 'Website', 'Search Keywords', 'Street Address']
        combined_features_df = pd.DataFrame([features1, features2]).reindex(columns=all_columns)

        num_cols = ['Latitude', 'Longitude', 'Number of Reviews']
        cat_cols = ['Name', 'Website', 'Search Keywords', 'Street Address']
        combined_features_df[num_cols] = combined_features_df[num_cols].fillna(0)
        combined_features_df[cat_cols] = combined_features_df[cat_cols].fillna("Unknown")

        X_transformed = preprocessor.transform(combined_features_df).toarray()

        for _ in range(5):  # Duplicate incorrect sample 5 times
            x_train.append(X_transformed[0])
            y_train.append(0)

    # Save the updated DataFrame to a CSV file
    cut_edges_df.to_csv(csv_file_path, index=False)

    if len(x_train) > 0:
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print("Model retraining started")
        model.train_on_batch(x_train, y_train)
        print("Model retrained.")
        model.save('trained_model.h5')
        print("Model saved.")
        cut_edges = []
        emit('retrain_status', {'status': 'complete'})  # Uncomment this if you're using some socket.emit function
        return True
    else:
        print("Not enough data for retraining.")
        return False

@socketio.on('request_retrain')
def handle_request_retrain(data=None):
    global business_features, cut_edges  # We need both for this part

    print(f"Debug: Received data = {data}")

    if data is None:
        print("Data not provided for retraining.")
        if len(cut_edges) > 0:
            attempt_and_announce_retrain()
        return

    node1 = data.get('node1', None)
    node2 = data.get('node2', None)

    if not node1 or not node2:
        print("Incomplete data for retraining.")
        return

    if node1 in business_features and node2 in business_features:
        cut_edges.append((node1, node2, business_features[node1], business_features[node2]))

    else:
        if node1 not in business_features:
            print(f"{node1} not found in business_features.")
        if node2 not in business_features:
            print(f"{node2} not found in business_features.")

if __name__ == '__main__':
    print("Application starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='KEY HERE', help='API Key')
    parser.add_argument('--radius', type=int, default=1000, help='Radius')
    parser.add_argument('--business_type', default='restaurant', help='Type of business')
    parser.add_argument('--limit', type=int, default=5, help='Limit')
    parser.add_argument('--filename', default="/Users/morgandixon/Desktop/businesses.csv", help='File containing business data')
    args = parser.parse_args()
    user_choice = input("Select an option: \n1. Use dataset \n2. Add More Data\n")
    if user_choice == "2":
        add_data_to_csv(filename, central_locations, general_collector)  # Assuming this function exists

    global gnn_processor
    gnn_processor = GNNProcessor(
        model=None,  # Removed args.model
        model_path=model_path,
        api_key=args.api_key,
        radius=args.radius,
        business_type=args.business_type,
        limit=args.limit,
        filename=args.filename
    )

    if not gnn_processor.model:
        try:
            print("Using existing trained model.")
            gnn_processor.model = load_model(model_path)
        except OSError as e:
            print(f"Error loading model: {e}. Retraining...")
            gnn_processor.model = gnn_processor.train_model()
            if gnn_processor.model is not None:
                gnn_processor.model.save(model_path)
            else:
                print("Model is None. Cannot save.")

        user_choice = input("Do you want to retrain the model? (y/n): ")
        if user_choice.lower() == 'y' or gnn_processor.model is None:
            print("Training model...")
            model = gnn_processor.train_model()
            if model is not None:
                model.save(model_path)
            else:
                print("Model is None. Cannot save.")
        else:
            try:
                print("Using existing trained model.")
                model = load_model(model_path)
                if model is not None:
                    model.save(model_path)
                else:
                    print("Model is None. Cannot save.")
            except OSError as e:
                print(f"Error loading model: {e}. Retraining...")
                model = gnn_processor.train_model()
                if model is not None:
                    model.save(model_path)
                else:
                    print("Model is None. Cannot save.")

    user_choice = input("Do you want to reset and retrain the model? (y/n): ")
    if user_choice.lower() == 'y' or gnn_processor.model is None:
        print("Resetting and retraining model...")
        gnn_processor.model = None  # Reset the model
        gnn_processor.preprocessor = None  # Reset the preprocessor
        model = gnn_processor.train_model()
        if model is not None:
            model.save(model_path)
        else:
            print("Model is None. Cannot save.")
    print("Application started")
    socketio.run(app, debug=False)
