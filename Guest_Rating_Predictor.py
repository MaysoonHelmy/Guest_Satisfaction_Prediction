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
from sklearn.metrics import r2_score

# nltk resources
nltk.download(['punkt', 'stopwords', 'wordnet'])

class GuestRatingPredictor:
    
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
        self.light_gray = "#F5F5F5"
        
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
                        text="Guest Rating Predictor",
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
        model_path = r"C:\Users\dell\Desktop\Guest_Satisfaction_Prediction\Milestone 1 Model Trials 2\CatBoost.pkl"
        encoder_path = r"C:\Users\dell\Desktop\Guest_Satisfaction_Prediction\Milestone 1 PreProcessing\label_encoder_model.pkl"

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.status_label.config(text="Model and encoder loaded successfully", fg="green")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.status_label.config(text="Data loaded successfully", fg="green")
            messagebox.showinfo("Success", "Data loaded successfully")

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
        
        text_columns = {'room_type', 'cancellation_policy'}
        for col in text_columns:
            df[col] = df[col].apply(self.preprocess_text)
        
        df.rename(columns={
            'room_type': 'room_type_cleaned', 
            'cancellation_policy': 'cancellation_policy_cleaned'
        }, inplace=True)
        
        def parse_amenities(amenities_str):
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
        binary_features_df = binary_features_df.astype(int)
        df = pd.concat([df, binary_features_df], axis=1)

        df['nightly_price'] = df['nightly_price'].replace(r'[\$,]', '', regex=True).astype(float)
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100
        df['host_listings_count'] = pd.to_numeric(df['host_listings_count'], errors='coerce').fillna(0)
        
        df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

        bool_mapping = {
            'instant_bookable': {'t': 1, 'f': 0},
            'host_identity_verified': {'t': 1, 'f': 0},
            'host_is_superhost': {'t': 1, 'f': 0}
        }
        
        mapping = {
            'within an hour': 1,
            'within a few hours': 2,
            'within a day': 3,
            'a few days or more': 4
        }       
        df['host_response_time'] = df['host_response_time'].map(mapping)
        
        for col, mp in bool_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mp)

        label_encode_cols = ['room_type_cleaned', 'cancellation_policy_cleaned']
        for col in label_encode_cols:
            if col in df.columns and col in self.encoder.classes_:
                df[col] = self.encoder.transform(df[col])
        
        df['review_duration_days'] = (df['last_review'] - df['first_review']).dt.days
        df['host_duration_days'] = (pd.to_datetime("today") - df['host_since']).dt.days
        df['price_value'] = df['nightly_price'] / df['accommodates']
        df['host_response_power'] = df['host_response_rate'] * np.log1p(df['host_listings_count'])
        df['host_commitment'] = df['host_duration_days'] / (df['number_of_stays'] + 1)
        df['bedroom_quality'] = df['bedrooms'] / (df['bathrooms'] + 0.5)
        df['space_per_guest'] = df['accommodates'] / (df['beds'] + 0.1)
        df['essential_amenities'] = df[['has_wifi', 'has_air_conditioning', 'has_kitchen']].sum(axis=1)
        df['review_consistency'] = df['number_of_reviews'] / (df['review_scores_rating'] + 1)
        
        return df

    def predict(self):
        if self.df is None or self.model is None:
            self.status_label.config(text="Please load data and ensure model is ready", fg="red")
            messagebox.showwarning("Warning", "Data or model not loaded")
            return

        self.status_label.config(text="Processing data...", fg=self.text_color)
        
        df_features = self.Feature_Engineering(self.df)
        
        all_features = [
            'host_response_rate', 'host_is_superhost', 'host_listings_count',
            'accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews',
            'number_of_stays', 'review_duration_days', 'host_duration_days',
            'price_value', 'host_response_power', 'host_commitment',
            'bedroom_quality', 'space_per_guest', 'essential_amenities',
            'review_consistency', 'instant_bookable', 'host_identity_verified',
            'room_type_cleaned', 'cancellation_policy_cleaned'
        ]
        
        X = df_features[all_features]
        self.df['Predicted Rating'] = self.model.predict(X)
        
        self.show_data(self.df[['name', 'description', 'Predicted Rating']])
        self.status_label.config(text="Predictions completed successfully", fg="green")

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

        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def load_actual_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
        
            actual_df = pd.read_csv(file_path)
            
            if "Predicted Rating" not in self.df.columns:
                messagebox.showerror("Error", 
                                    "No predictions found. Please use the 'Predict' button before loading Actual data.")
                return

            self.df['review_scores_rating'] = actual_df['review_scores_rating']

            # Calcualate R² Score
            r2 = r2_score(self.df['review_scores_rating'], self.df['Predicted Rating'])
            self.status_label.config(text=f"R² computed: {r2:.4f}", fg="green")
            messagebox.showinfo("Regression Metric", f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1960x900")
    app = GuestRatingPredictor(root)
    root.mainloop()