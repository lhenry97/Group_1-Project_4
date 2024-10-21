# pip install psycopg2
# pip install numpy

# Ignore SQLITE warnings related to Decimal numbers in the Chinook database
import warnings
warnings.filterwarnings('ignore')

# Import Dependencies
import numpy as np
from flask import Flask, jsonify, render_template
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy import func
from config import password
import psycopg2


#################################################
# Database Setup
#################################################
# Create engine using the `postgres` database file on local host
engine = create_engine(f'postgresql://postgres:{password}@localhost:5432/Cancer_db')
connection = engine.connect()

# Reflect Database into ORM classes
Base = automap_base()
Base.prepare(autoload_with=engine)
Base.classes.keys()

# Define the tables
cd = Base.classes.Cancer_Data

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Flask Routes
#################################################

@app.route("/")
def index():
    print("Server received request for 'Home/Index' page...")
    return render_template("index.html")

@app.route("/Zahra_LR")
def logistic_regression():
    print("Server received request for 'Zahra's Logistic Regression Model' page...")
    return render_template("lin_reg_model.html")

@app.route('/Jess_Aria_SVM')
def SVM():
    print("Server received request for 'Jess' and Aria's SVM Model' page...")  
    return render_template("SVM.html")

@app.route('/Jess_Aria_RF')
def RF():
    print("Server received request for 'Jess' and Aria's RF Model' page...")  
    return render_template("ran_for_model.html")

@app.route('/Cathy_NN_KT')
def KT():
    print("Server received request for 'Cathy's KT Model' page...")  
    return render_template("ker_tun_model.html")

@app.route('/Lauren_Eval')
def evaluation():
    print("Server received request for 'Lauren Evaluatiion of Models' page...")  
    return render_template("eval.html")

@app.route('/Prediction')
def prediction():
    print("Server received request for 'Lauren Evaluatiion of Models' page...")  
    return render_template("prediction_app.html")

@app.route('/viz1')
def visualisations():
    print("Server received request for 'Lauren Evaluatiion of Models' page...")  
    return render_template("vizs.html")

@app.route("/db_data")
def get_cancer_db_data():
    print("Server received request for fetched 'Jsonified database data'...")
    session = Session(engine)

    try:
        query = text("""
        SELECT json_agg(row_to_json("Cancer_Data")) FROM "Cancer_Data";
        """)

        # Execute the query
        results = session.execute(query)
        # all_data = results.scalar()  # Get the first column of the first row
    
        # # Convert list of tuples into normal list
        # all_data = list(np.ravel(results))
        all_data = results.fetchone()[0]  # Get the first row and first column
        return jsonify(all_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

if __name__ == '__main__':
    app.run(debug=True)