import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from collections import Counter

# nltk resources
nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)

class GuestSatisfactionPredictor:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Guest Rating Predictor")

        self.filename = None
        self.df = None
        self.model = None
        self.encoder = None

        self.bg_color = "#FFFFFF"
        self.primary_color = "#2196F3"
        self.secondary_color = "#4CAF50"
        self.accent_color = "#FF5722"
        self.text_color = "#333333"
        self.light_gray =  "#F5F5F5"
        
        self.root.configure(bg=self.bg_color)
        self.style_config()
        self.build_ui()
        self.load_model_and_encoder()

    def style_config(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview",
                        background=self.light_gray,
                        foreground=self.text_color,
                        rowheight=25,
                        fieldbackground=self.light_gray)
        style.configure("Treeview.Heading",
                        background=self.primary_color,
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading",
                background=[('active', self.primary_color)])

    def build_ui(self):
        container = tk.Frame(self.root, bg=self.bg_color)
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        top_frame = tk.Frame(container, bg=self.bg_color)
        top_frame.pack(fill=tk.X, pady=(0, 20))

        title_frame = tk.Frame(top_frame, bg=self.primary_color)
        title_frame.pack(fill=tk.X)
        
        title = tk.Label(title_frame, 
                        text="Guest Satisfaction Predictor",
                        font=("Helvetica", 28, "bold"),
                        bg=self.primary_color,
                        fg="white",
                        pady=20)
        title.pack()

        control_frame = tk.Frame(container, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        btn_load = tk.Button(control_frame,
                            text="Load Data",
                            command=self.load_data,
                            font=("Helvetica", 12),
                            bg=self.primary_color,
                            fg="white",
                            pady=10,
                            padx=20,
                            border=0,
                            cursor="hand2")
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_predict = tk.Button(control_frame,
                                text="Predict",
                                command=self.predict,
                                font=("Helvetica", 12),
                                bg=self.secondary_color,
                                fg="white",
                                pady=10,
                                padx=20,
                                border=0,
                                cursor="hand2")
        btn_predict.pack(side=tk.LEFT, padx=5)

        btn_load_actual = tk.Button(control_frame,
                                    text="Load Actual Data & Calc Accuracy",
                                    command=self.load_actual_data,
                                    font=("Helvetica", 12),
                                    bg=self.accent_color,
                                    fg="white",
                                    pady=10,
                                    padx=20,
                                    border=0,
                                    cursor="hand2")
        btn_load_actual.pack(side=tk.LEFT, padx=5)

        self.status_frame = tk.Frame(container, bg=self.bg_color)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(self.status_frame,
                                    text="Ready to load data...",
                                    font=("Helvetica", 10),
                                    bg=self.bg_color,
                                    fg=self.text_color)
        self.status_label.pack(anchor='w')

        tree_container = tk.Frame(container, 
                                bg=self.primary_color,
                                padx=2,
                                pady=2)
        tree_container.pack(fill=tk.BOTH, expand=True)

        tree_frame = tk.Frame(tree_container, bg=self.bg_color)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')

        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        self.result_frame = tk.Frame(container, bg=self.bg_color)
        self.result_frame.pack(fill=tk.X, pady=(20, 0))
    
    def load_model_and_encoder(self):
        try:
            model_path = "Milestone 2 GUI/xgboost_pipeline.pickle"
            encoder_path = "Milestone 2 GUI/label_encoders.pkl"
            y_encoder_path = "Milestone 2 GUI/y_label_encoder.pkl"

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
            with open(y_encoder_path, "rb") as f:
                self.y_encoder = pickle.load(f)

            self.status_label.config(text="Model and encoder loaded successfully", fg="green")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Failed to load model or encoder: {str(e)}")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.status_label.config(text="Data loaded successfully", fg="green")
                messagebox.showinfo("Success", "Data loaded successfully")
                display_cols = ['name', 'description']
                display_cols = [col for col in display_cols if col in self.df.columns]
                if display_cols:
                    self.show_data(self.df[display_cols].head(10))
                else:
                    self.show_data(self.df.head(10))
            except Exception as e:
                self.status_label.config(text=f"Error loading data: {str(e)}", fg="red")
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text, language='english')
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)

    def Feature_Engineering(self, df):
        df = df.copy()
        
        text_columns = {'neighbourhood_cleansed', 'instant_bookable','cancellation_policy', 'property_type', 'room_type'}
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self.preprocess_text)
        
        if 'amenities' in df.columns:
            def parse_amenities(amenities_str):
                if not isinstance(amenities_str, str):
                    return []
                amenities_str = amenities_str.strip('{}')
                return [item.strip().strip('"') for item in amenities_str.split(',')]

            df['amenities_parsed'] = df['amenities'].apply(parse_amenities)

            amenity_categories = {
                'Basic Necessities': {'Heating', 'Hot water', 'Smoke detector', 'Carbon monoxide detector'},
                'Comfort/Convenience': {'Air conditioning', 'Kitchen', 'Iron', 'Laptop friendly workspace', 'Essentials', 'Hangers'},
                'Technology': {'TV', 'Wifi'},
                'Safety': {'Smoke detector', 'Carbon monoxide detector', 'Lock on bedroom door'},
                'Recreation': {'Pool', 'Gym', 'Hot tub'},
                'Parking': {'Free parking on premises'},
                'Accessibility': {'Elevator', 'Private entrance'}
            }

            def create_binary_features(amenities_list):
                binary_features = {}
                all_amenities = set(amenities_list)
                for category, items in amenity_categories.items():
                    for item in items:
                        binary_features[f'has_{item.lower().replace(" ", "_")}'] = 1 if item in all_amenities else 0
                return binary_features

            binary_features_df = df['amenities_parsed'].apply(create_binary_features).apply(pd.Series)
            binary_features_df = binary_features_df.fillna(0).astype(int)
            df = pd.concat([df, binary_features_df], axis=1)

        # Convert price columns
        price_columns = ['nightly_price', 'cleaning_fee', 'security_deposit', 'extra_people']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        if 'host_response_rate' in df.columns:
            df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '')
            df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0) / 100
            
        if 'host_listings_count' in df.columns:
            df['host_listings_count'] = pd.to_numeric(df['host_listings_count'], errors='coerce').fillna(0)
        
        # Convert date columns
        date_cols = ['first_review', 'last_review', 'host_since']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Boolean mapping
        bool_mapping = {
            'instant_bookable': {'t': 1, 'f': 0},
            'host_identity_verified': {'t': 1, 'f': 0},
            'host_is_superhost': {'t': 1, 'f': 0}
        }
        
        # Response time mapping
        if 'host_response_time' in df.columns:
            mapping = {
                'within an hour': 1,
                'within a few hours': 2,
                'within a day': 3,
                'a few days or more': 4
            }
            df['host_response_time'] = df['host_response_time'].map(mapping)
        
        # Apply boolean mappings
        for col, mp in bool_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mp).fillna(0).astype(int)
        
        # Label encoding
        label_encode_cols = ['room_type', 'cancellation_policy','property_type','host_identity_verified']
        for col in label_encode_cols:
            if col in df.columns and col in self.encoder:
                df[col] = self.encoder[col].transform(df[col].astype(str))
        
        # Process percentage columns
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].astype(str).str.contains('%').any():
                    try:
                        df[col] = df[col].str.replace('%', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 100
                    except Exception:
                        pass
        
        # Feature creation
        if all(col in df.columns for col in ['first_review', 'last_review']):
            df['review_duration_days'] = (df['last_review'] - df['first_review']).dt.days.fillna(0)
        
        if 'host_since' in df.columns:
            df['host_duration_days'] = (pd.to_datetime("today") - df['host_since']).dt.days.fillna(0)
        
        if all(col in df.columns for col in ['nightly_price', 'accommodates']):
            df['price_value'] = df['nightly_price'] / df['accommodates'].replace(0, 1)
        
        if all(col in df.columns for col in ['host_response_rate', 'host_listings_count']):
            df['host_response_power'] = df['host_response_rate'] * np.log1p(df['host_listings_count'])
        
        if all(col in df.columns for col in ['host_duration_days', 'number_of_stays']):
            df['host_commitment'] = df['host_duration_days'] / (df['number_of_stays'].fillna(0) + 1)
        
        if all(col in df.columns for col in ['bedrooms', 'bathrooms']):
            df['bedroom_quality'] = df['bedrooms'] / (df['bathrooms'] + 0.5)
        
        if all(col in df.columns for col in ['accommodates', 'beds']):
            df['space_per_guest'] = df['accommodates'] / (df['beds'].fillna(1) + 0.1)
        
        amenity_columns = ['has_wifi', 'has_air_conditioning', 'has_kitchen']
        if all(col in df.columns for col in amenity_columns):
            df['essential_amenities'] = df[amenity_columns].sum(axis=1)
        
        return df
    
    def predict(self):
        if self.df is None or self.model is None:
            self.status_label.config(text="Please load data and ensure model is ready", fg="red")
            messagebox.showwarning("Warning", "Data or model not loaded")
            return
        try:
            self.status_label.config(text="Processing data...", fg=self.text_color)

            df_feats = self.Feature_Engineering(self.df)
            
            # Define required features based on the model
            numeric_features = [
                'host_response_rate', 'host_is_superhost', 'accommodates',
                'bathrooms', 'bedrooms', 'beds', 'nightly_price',
                'cleaning_fee', 'number_of_reviews', 'minimum_nights',
                'host_listings_count', 'number_of_stays', 'security_deposit',
                'extra_people', 'maximum_nights', 'guests_included'
            ]

            categorical_features = [
                'neighbourhood_cleansed', 'instant_bookable',
                'cancellation_policy', 'property_type', 'room_type'
            ]

            amenity_features = [
                'has_wifi', 'has_air_conditioning', 'has_kitchen',
                'has_heating', 'has_tv', 'has_free_parking_on_premises',
                'has_iron', 'has_laptop_friendly_workspace'
            ]
            
            # Filter to only include features that exist in the dataframe
            all_feats = [f for f in numeric_features + categorical_features + amenity_features if f in df_feats.columns]
            
            # Check if we have enough features
            if len(all_feats) < 5:
                self.status_label.config(text="Not enough features in the data", fg="red")
                messagebox.showerror("Error", "The dataset doesn't contain enough required features")
                return

            X = df_feats[all_feats]

            codes = self.model.predict(X)
            labels = self.y_encoder.inverse_transform(codes) 
            self.df['Guest Satisfaction'] = labels

            # Display results - only name, description and Guest Satisfaction
            display_cols = ['name', 'description', 'Guest Satisfaction']
            available_cols = [col for col in display_cols if col in self.df.columns]
            
            if len(available_cols) < 2:
                # If name or description are missing, use the first column and Guest Satisfaction
                available_cols = [self.df.columns[0], 'Guest Satisfaction']
            
            self.show_data(self.df[available_cols])
            self.status_label.config(text="Predictions completed successfully", fg="green")
            
        except Exception as e:
            self.status_label.config(text=f"Error during prediction: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Failed to make predictions: {str(e)}")

    def show_data(self, df):
        self.tree.delete(*self.tree.get_children())
        
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col.upper())
            max_width = max(len(str(col)), df[col].astype(str).str.len().max() if len(df) > 0 else 10)
            display_width = min(max(max_width * 7, 100), 300)
            self.tree.column(col, anchor="center", width=display_width)

        for i, row in enumerate(df.iterrows()):
            values = row[1].tolist()
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            self.tree.insert("", "end", values=values, tags=(tag,))

        self.tree.tag_configure('oddrow', background='#FFFFFF')
        self.tree.tag_configure('evenrow', background='#F5F5F5')

    def load_actual_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No data loaded. Please load data first.")
            return
            
        if "Guest Satisfaction" not in self.df.columns:
            messagebox.showerror("Error", 
                            "No predictions found. Please use the 'Predict' button before loading Actual data.")
            return
        
        if not hasattr(self, 'y_encoder') or self.y_encoder is None:
            messagebox.showerror("Error", "Label encoder not initialized. Please train or load a model first.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                actual_df = pd.read_csv(file_path)
                
                if 'guest_satisfaction' not in actual_df.columns:
                    messagebox.showerror("Error", "The loaded file does not contain a 'guest_satisfaction' column.")
                    return
                
                # Add actual ratings to dataframe
                min_rows = min(len(self.df), len(actual_df))
                self.df['Actual Rating'] = actual_df['guest_satisfaction'].iloc[:min_rows].values
                
                # Convert labels using y_encoder
                actual_labels = actual_df['guest_satisfaction'].astype(str).iloc[:min_rows]
                pred_labels = self.df['Guest Satisfaction'].astype(str).iloc[:min_rows]

                # Check for unknown labels
                unknown_actual = [lbl for lbl in actual_labels.unique() if lbl not in self.y_encoder.classes_]
                unknown_pred = [lbl for lbl in pred_labels.unique() if lbl not in self.y_encoder.classes_]
                
                if unknown_actual:
                    messagebox.showerror("Error", f"Unknown actual labels: {', '.join(unknown_actual)}")
                    return
                if unknown_pred:
                    messagebox.showerror("Error", f"Unknown predicted labels: {', '.join(unknown_pred)}")
                    return

                # Transform to numeric values
                actual_numeric = self.y_encoder.transform(actual_labels)
                pred_numeric = self.y_encoder.transform(pred_labels)

                # Calculate accuracy
                acc = accuracy_score(actual_numeric, pred_numeric)
                correct = sum(actual_numeric == pred_numeric)
                total = len(actual_numeric)

                # Count label distribution
                actual_counts = Counter(actual_labels)
                pred_counts = Counter(pred_labels)

                # Prepare accuracy report
                accuracy_details = [
                    f"Overall Accuracy: {acc:.4f}",
                    f"Correct Predictions: {correct} out of {total}",
                    "\nActual Rating Distribution:"
                ]
                
                # Add class-wise counts
                for class_label in self.y_encoder.classes_:
                    accuracy_details.append(
                        f"  {class_label}: {actual_counts.get(class_label, 0)} actual / {pred_counts.get(class_label, 0)} predicted"
                    )

                # Update UI
                self.status_label.config(text=f"Accuracy: {acc:.4f}", fg="green")
                messagebox.showinfo("Accuracy Metrics", "\n".join(accuracy_details))
                
                # Display both ratings
                display_cols = ['name', 'description', 'Guest Satisfaction', 'Actual Rating']
                available_cols = [col for col in display_cols if col in self.df.columns]
                self.show_data(self.df[available_cols])

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status_label.config(text=f"Error processing data: {str(e)}", fg="red")
                messagebox.showerror("Error", f"Data processing failed: {str(e)}")
            
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1960x900")
    app = GuestSatisfactionPredictor(root)
    root.mainloop()