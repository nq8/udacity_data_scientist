# Disaster Response Pipeline Project
This projects consists of three steps:
1) Clean and prepare raw data using an ETL pipeline
2) Train a model on the cleaned data
3) Run the web app to visualize graphs of the training data and enable the user to categorize a 
message based on the trained model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Structure:
```
project
│   README.md 
│
└───app
│   │   run.py -> File to start the web app
│   │
│   └───templates
│       │   go.html -> Handling a new message input in web visualization
|       |   master.html -> Used for web visualization
│   
└───data
|   │   disaster_categories.csv -> raw data used in process_data.py
|   │   disaster_messages.csv -> raw data used in process_data.py
|   │   process_data.py -> Prepare and save data
│   
└───models
    │   train_classifier.py -> Train model on pre-created database
```
