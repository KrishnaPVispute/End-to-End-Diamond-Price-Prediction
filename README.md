💎 Diamond Price Prediction

A Machine Learning project that predicts the price of diamonds based on features such as carat, cut, color, clarity, and dimensions. The project uses scikit-learn models, a training pipeline, and a Streamlit web app for easy interaction.

📌 Project Workflow

Data Ingestion – Load and preprocess the dataset.

Data Transformation – Apply scaling and encoding.

Model Training – Train multiple ML models and select the best one.

Prediction Pipeline – Make predictions using the saved model & preprocessor.

Streamlit App – Simple UI for users to input features and predict diamond prices.

🚀 Tech Stack

Python 3.9+

Pandas, NumPy

scikit-learn (ML models)

Streamlit (Web App)

Pickle (Model & Preprocessor saving)


📂 Project Structure
diamond-price-prediction/
│
├── src/
│   ├── components/          # Data ingestion, transformation, model training
│   ├── pipeline/            # Training & prediction pipeline
│   ├── utils.py             # Helper functions
│   ├── logger.py            # Logging setup
│   └── exception.py         # Custom exception handling
│
├── artifacts/               # Saved model & preprocessor
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── app.py                   # Streamlit web app
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup
└── README.md                # Project documentation

🏆 Features

✅ Multiple regression models compared (Linear, Ridge, Lasso, ElasticNet, DecisionTree).
✅ Automated model evaluation & best-model selection.
✅ Scalable & modular pipeline.
✅ User-friendly Streamlit UI.
✅ Easy deployment-ready structure.


🤝 Contributing
Pull requests are welcome! If you find a bug or want to add a feature, feel free to open an issue.

📜 License

This project is licensed under the MIT License.

⚡ Author: https://github.com/KrishnaPVispute
