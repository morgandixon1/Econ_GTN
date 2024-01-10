import numpy as np
import itertools
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from keras.models import load_model 
import csv 
from geopy.distance import geodesic
from random import sample
import math
from tensorflow.keras.callbacks import EarlyStopping
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from tensorflow.keras.preprocessing.sequence import pad_sequences

class GNNProcessor:
    def __init__(self, businesses_df):
        self.businesses_df = businesses_df

    def process_data(self):
        connected_edges = {}
        disconnected_edges = {}

        businesses_list = "/Users/morgandixon/Desktop/businesses3.5.csv"
        cut_edges_list = "cut_edges.csv"
        good_edges_list = "good_edges.csv"

        try:
            businesses_df = pd.read_csv(businesses_list)
    
            print(f"Total businesses in list: {len(businesses_df)}")

            for col in ['Latitude', 'Longitude']:
                businesses_df[col].fillna(-999, inplace=True)

            string_cols = businesses_df.select_dtypes(include=['object']).columns
            for col in string_cols:
                businesses_df[col].fillna("Placeholder", inplace=True)

            int_cols = businesses_df.select_dtypes(include=['int']).columns
            for col in int_cols:
                businesses_df[col].fillna(-1, inplace=True)

            def get_full_row(lat, long):
                matching_row = businesses_df[
                    (businesses_df['Latitude'] == lat) & (businesses_df['Longitude'] == long)
                ]
                if not matching_row.empty:
                    row = matching_row.iloc[0]
                    # Assuming 'Name' is a unique identifier
                    return row['Name'], '|'.join(row.astype(str))
                else:
                    return None, None

            cut_edges_df = pd.read_csv(cut_edges_list)
            for _, row in cut_edges_df.iterrows():
                node1_id, node1_full = get_full_row(row['Latitude1'], row['Longitude1'])
                node2_id, node2_full = get_full_row(row['Latitude2'], row['Longitude2'])

                if node1_id is None or node2_id is None:
                    continue

                disconnected_edges[(node1_id, node2_id)] = (node1_full, node2_full)

            print(f"Total pairs in disconnected_edges: {len(disconnected_edges)}")
            example_key = next(iter(disconnected_edges))
            # Update connected_edges
            good_edges_df = pd.read_csv(good_edges_list)
            # In your process_data function:
            for _, row in cut_edges_df.iterrows():
                node1_id, node1_full = get_full_row(row['Latitude1'], row['Longitude1'])
                node2_id, node2_full = get_full_row(row['Latitude2'], row['Longitude2'])

                if node1_id is None or node2_id is None:
                    continue

                connected_edges[(node1_id, node2_id)] = (node1_full, node2_full)

            print(f"Total pairs in connected_edges: {len(connected_edges)}")
            example_key = next(iter(connected_edges))

            self.businesses_df = businesses_df 
        except FileNotFoundError as e:
            print(f"Could not find the file: {e.filename}")

        return connected_edges, disconnected_edges, businesses_df

    def one_hot_encode_data(self, data, max_length, vocab_size):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)

        # Debugging: Check the sequence lengths
        print("Sample sequence:", sequences[0])
        print("Max sequence length:", max(len(seq) for seq in sequences))

        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

        # One-hot encoding
        one_hot = np.zeros((len(padded_sequences), max_length, vocab_size))
        for i, sequence in enumerate(padded_sequences):
            for j, index in enumerate(sequence):
                if 0 < index < vocab_size:
                    one_hot[i, j, index] = 1

        # Debugging: Check the shape of the one-hot encoded data
        print("Shape of one-hot encoded data:", one_hot.shape)

        return one_hot

    def train_connectivity_model(self, connected_edges, disconnected_edges, model_save_path='connectivity.keras', update=False):
        print("Starting training of connectivity model...")

        # Data preparation for nodes
        connected_nodes = [(node1, node2) for node1, node2 in connected_edges]
        disconnected_nodes = [(node1, node2) for node1, node2 in disconnected_edges]

        all_edges = connected_nodes + disconnected_nodes
        edge_labels = [1] * len(connected_nodes) + [0] * len(disconnected_nodes)
        combined = list(zip(all_edges, edge_labels))
        np.random.shuffle(combined)
        all_edges, edge_labels = zip(*combined)

        # Separate the nodes again for one-hot encoding
        all_nodes = [node for edge in all_edges for node in edge]
        all_labels = [label for edge_label in edge_labels for label in (edge_label, edge_label)]

        # Determine the maximum length of the sequences and the vocabulary size
        max_length = min(max(len(node.split()) for node in all_nodes), 256)
        vocab_size = len(set(token for node in all_nodes for token in node.split()))

        split_index = int(len(all_nodes) * 0.75)
        train_nodes, val_nodes = all_nodes[:split_index], all_nodes[split_index:]
        train_labels, val_labels = all_labels[:split_index], all_labels[split_index:]

        # One-hot encode node features for training and validation sets
        train_node_features = self.one_hot_encode_data(train_nodes, max_length, vocab_size)
        val_node_features = self.one_hot_encode_data(val_nodes, max_length, vocab_size)

        # Debugging: Check the shapes of the features and labels
        print("Shape of train_node_features:", train_node_features.shape)
        print("Shape of train_labels:", np.array(train_labels).shape)
        print("Shape of val_node_features:", val_node_features.shape)
        print("Shape of val_labels:", np.array(val_labels).shape)

        # Creating TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_node_features, np.array(train_labels))).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_node_features, np.array(val_labels))).batch(32)

        # Model architecture
        if update and self.agent_model is not None:
            connectivity_model = tf.keras.models.load_model(model_save_path)
        else:
            inputs = layers.Input(shape=(max_length, vocab_size))

            # Self-attention layers
            attention_output = layers.MultiHeadAttention(num_heads=6, key_dim=64)(inputs, inputs)
            attention_output = layers.LayerNormalization()(attention_output)  # Adding normalization

            # Bidirectional LSTM layer
            lstm_output = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(attention_output)

            # Flatten and feed into dense layers
            flat_output = layers.Flatten()(lstm_output)
            dense_output = layers.Dense(64, activation='relu')(flat_output)
            dropout_output = layers.Dropout(0.5)(dense_output)
            outputs = layers.Dense(1, activation='sigmoid')(dropout_output)

            connectivity_model = models.Model(inputs=inputs, outputs=outputs)
            connectivity_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                                    loss='binary_crossentropy', metrics=['accuracy'])

        # Callbacks and training
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
        history = connectivity_model.fit(train_dataset, validation_data=val_dataset, epochs=45, callbacks=[early_stopping, reduce_lr])
        eval_results = connectivity_model.evaluate(val_dataset)

        # Output and saving
        print(f"Training complete. Model metrics: {eval_results}")
        connectivity_model.save(model_save_path)

        return connectivity_model, eval_results, history


    def train_egrp_model(self, filepath):
        h2o.init()
        
        print("Training model manually with H2O GBM")
        businesses_df = pd.read_csv(filepath)
        businesses_df = businesses_df[businesses_df['Estimated EGRP'].notna()]
        print(f"Columns considered in the model: {businesses_df.columns.tolist()}")
        h2o_df = h2o.H2OFrame(businesses_df)
        y = 'Estimated EGRP'
        X = [name for name in h2o_df.columns if name != y]
        
        train, test = h2o_df.split_frame([0.8])
        gbm = H2OGradientBoostingEstimator(
            model_id='egrp',  # Set the model_id here
            ntrees=50,
            max_depth=6,
            learn_rate=0.1,
            stopping_tolerance=0.01,
            seed=9,
            nfolds=5,
            fold_assignment='Modulo',
            keep_cross_validation_predictions=True
        )

        gbm.train(x=X, y=y, training_frame=train)
        print("Model Summary")
        print(gbm.summary())
        print("Variable Importance")
        print(gbm.varimp())
        performance = gbm.model_performance(test)
        print("Model Performance on Test Data")
        print(performance)
        preds = gbm.predict(test)
        preds_df = h2o.as_list(preds)
        preds_df[preds_df < 0] = 0
        y_test_df = h2o.as_list(test[y])
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test_df)), y_test_df, color='blue', label='Actual')
        plt.scatter(range(len(preds_df)), preds_df, color='red', label='Predicted')
        plt.title('Actual vs Predicted EGRP')
        plt.xlabel('Index')
        plt.ylabel('EGRP Value')
        plt.legend()
        plt.show()
        
        # Save the model
        model_path = h2o.save_model(model=gbm, force=True)
        print(f"The model is saved at: {model_path}")

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def find_close_businesses(filename, latitude, longitude, keyword, radius_miles):
    print("find_close_businesses function started.")  # New print statement
    businesses = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_float(row['Latitude']) and is_float(row['Longitude']):
                business = {
                    'name': row['Name'],
                    'coords': (float(row['Latitude']), float(row['Longitude'])),
                    'Website': row.get('Website', ''),
                    'Search Keywords': row.get('Search Keywords', ''),
                    'Street Address': row.get('Street Address', ''),
                    'Number of Reviews': row.get('Number of Reviews', ''),
                    'Estimated EGRP': row.get('Estimated EGRP', None)  # Added this line
                }
                businesses.append(business)
            else:
                print(f"Skipping row due to invalid data: {row}")

    if not businesses:
        print("No valid businesses found.")
        return []

    central_coords = (latitude, longitude)
    close_businesses = [business for business in businesses
                        if geodesic(central_coords, business['coords']).miles <= radius_miles]

    if close_businesses:
        print(f"Found {len(close_businesses)} businesses within {radius_miles} miles")
        for i, business in enumerate(close_businesses, 1):
            print(f"{i}. {business['name']} ({business['coords']})")
    else:
        print("Not enough close businesses found.")

    print("find_close_businesses function completed.")  # New print statement
    return close_businesses

relationship_value_sum = {}  # A dictionary to store the sum of relationship values for each input node

def calculate_relationship_value(relationship_data_dict, pair_key, businesses_df):
    global relationship_value_sum  # Declare the variable as global to update it within the function
    
    print(f"Checking relationship for pair: {pair_key}")

    if pair_key not in relationship_data_dict:
        print(f"Pair {pair_key} not found in relationship_data_dict.")
        return None, None

    data = relationship_data_dict[pair_key]
    S = data['connectivity_score']
    input_node_name = data['input_node']
    given_node_name = data['given_node']

    input_row = businesses_df[businesses_df['Name'] == input_node_name].iloc[0]
    given_row = businesses_df[businesses_df['Name'] == given_node_name].iloc[0]

    EGRP_input = input_row['Estimated EGRP'] if input_row['Estimated EGRP'] != 0 else None
    EGRP_given = given_row['Estimated EGRP'] if given_row['Estimated EGRP'] != 0 else None

    NAICS_input = input_row['NAICS']
    NAICS_given = given_row['NAICS']

    print(f"NAICS of input node: {NAICS_input}")
    print(f"NAICS of given node: {NAICS_given}")

    if EGRP_input is None or EGRP_given is None:
        print("EGRP_input or EGRP_given is zero or None. Returning None values.")
        return None, None

    print(f"S (connectivity score): {S}")
    print(f"EGRP_input: {EGRP_input}")
    print(f"EGRP_given: {EGRP_given}")

    M = 1  # Placeholder for industry-based multiplier
    w1, w2 = 0.5, 0.5  # Weights for the calculation
    
    V = M * S * (w1 * EGRP_input + w2 * EGRP_given)
    
    if not np.isnan(V):
        if input_node_name in relationship_value_sum:
            relationship_value_sum[input_node_name] += V
        else:
            relationship_value_sum[input_node_name] = V
    
    V_normalized = (EGRP_input / relationship_value_sum[input_node_name]) * V if relationship_value_sum[input_node_name] != 0 else 0
    
    total_EGRP = EGRP_input + EGRP_given
    V_percentage = (V_normalized / total_EGRP) * 100 if total_EGRP != 0 else 0

    expected_expenditure = 10  # Placeholder for expenditure check based on industry norms
    if V_percentage > expected_expenditure:
        print("Warning: V_percentage exceeds expected expenditure for this industry.")

    print(f"Calculated V: {V_normalized}")
    print(f"Calculated V_percentage: {V_percentage}%")

    return V_normalized, V_percentage

def make_predictions(form_data=None):
    if form_data is None:
        form_data = request.form.to_dict()
    print("Received form data:", form_data)

    # Initialize variables
    global node_data
    node_data = []
    edge_data = []
    relationship_data_dict = {}
    naics = None  # Initialize naics to None
    # Load models and data
    connectivity_model = load_model('connectivity.keras')
    businesses_df = pd.read_csv('/Users/morgandixon/Desktop/businesses3.5.csv')
    print("Columns in businesses_df:", businesses_df.columns)
    # Extract form data
    input_node = form_data.get('name', None)
    latitude_str = form_data.get('latitude', None)
    longitude_str = form_data.get('longitude', None)
    radius_str = form_data.get('radius', '2')

    # Try to convert latitude, longitude, and radius to float
    try:
        latitude = float(latitude_str)
        longitude = float(longitude_str)
        radius_miles = float(radius_str)
    except (ValueError, TypeError):
        latitude = longitude = radius_miles = None

    # If no form data for name, latitude, or longitude, choose a random business
    if not all([input_node, latitude, longitude]):
        rand_index = random.randint(0, len(businesses_df) - 1)
        random_business = businesses_df.iloc[rand_index].to_dict()
        input_node = random_business['Name']
        latitude = float(random_business['Latitude'])
        longitude = float(random_business['Longitude'])
        naics = random_business.get('NAICS')
        radius_miles = 2.0

    filename = '/Users/morgandixon/Desktop/businesses3.5.csv'
    close_businesses = find_close_businesses(filename, latitude, longitude, "", radius_miles if radius_miles is not None else 2.0)

    if not close_businesses:
        print("No close businesses found.")
        return {"status": "FAIL", "message": "No close businesses found."}

    node_data.append({
        'name': input_node,
        'latitude': latitude,
        'longitude': longitude,
        'naics': naics
    })
    print(f"Input Node: {input_node}")
    for i, given_node in enumerate(close_businesses):
        if i >= 5:  # Limit to 5 predictions
            break
        given_node_name = given_node.get('name')
        concatenated_data = str(input_node) + "||" + str(given_node_name)
        concatenated_data = np.array([concatenated_data], dtype=np.float32)

        connectivity_prediction = connectivity_model.predict(concatenated_data)[0][0]
        print(f"Raw Connectivity Score for {given_node_name}: {connectivity_prediction}")

        coords = given_node.get('coords')
        if coords is None:
            print(f"Warning: Missing coordinates for given_node={given_node_name}. Skipping this edge.")
            continue

        given_node_latitude, given_node_longitude = coords
        node_data.append({
            'name': given_node_name,
            'latitude': given_node_latitude,
            'longitude': given_node_longitude,
            'naics': naics
        })

        edge_data.append({
            'input_node': input_node,
            'given_node': given_node_name,
            'connectivity_score': float(connectivity_prediction),
            'input_node_latitude': latitude,
            'input_node_longitude': longitude,
            'given_node_latitude': given_node_latitude,
            'given_node_longitude': given_node_longitude
        })
        relationship_data_dict[f"{input_node}_{given_node_name}"] = {
            'input_node': input_node,
            'given_node': given_node_name,
            'connectivity_score': float(connectivity_prediction),
            'input_node_data': {
                'latitude': latitude,
                'longitude': longitude,
                'naics': naics,
                # Add other data you might have here
            },
            'given_node_data': {
                'latitude': given_node_latitude,
                'longitude': given_node_longitude,
                'naics': naics,
                # Add other data you might have here
            }
        }

        V, V_percentage = calculate_relationship_value(relationship_data_dict, f"{input_node}_{given_node_name}", businesses_df)

    return {"status": "OK", "node_data": node_data, "edge_data": edge_data, "relationship_data_dict": relationship_data_dict}

if __name__ == '__main__':
    # Initialize processor and data
    business_data_df = pd.read_csv("/Users/morgandixon/Desktop/businesses3.5.csv")
    gnn_processor = GNNProcessor(business_data_df)
    connected_edges, disconnected_edges, businesses_df = gnn_processor.process_data()
    user_choice = input("Select the operation:\n1. Train EGRP Model\n2. Train Connectivity Model\n3. Calculate Relationship Value\nYour choice: ")

    if user_choice == '1':
        print("Training EGRP model...")
        gnn_processor.train_egrp_model("/Users/morgandixon/Desktop/businesses3.5.csv")

    elif user_choice == '2':
        print("Training connectivity model...")
        gnn_processor.connectivity_model = gnn_processor.train_connectivity_model(connected_edges, disconnected_edges)

    elif user_choice == '3':
        print("Calculating relationship value...")
        business_data_df = pd.read_csv("/Users/morgandixon/Desktop/businesses3.5.csv")
        rand_index = random.randint(0, len(business_data_df) - 1)
        random_business = business_data_df.iloc[rand_index].to_dict()
        input_node = random_business['Name']
        latitude_str = str(random_business['Latitude'])
        longitude_str = str(random_business['Longitude'])
        radius_str = "2"  # You can keep this as a constant or make it random too
        form_data = {
            'name': input_node,
            'latitude': latitude_str,
            'longitude': longitude_str,
            'radius': radius_str
        }
        
        result = make_predictions(form_data)

    else:
        print("Invalid choice. Exiting.")