from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Define the input features and target variable
X = [['John'], ['Jane'], ['Jack'], ['Jill']]
y = [25, 30, 22, 28]

# Encode the names as numeric values
le = LabelEncoder()
X_encoded = le.fit_transform([name[0] for name in X])

# Create the model
model = DecisionTreeRegressor()

# Fit the model to the data
model.fit(X_encoded.reshape(-1, 1), y)

# Save the model to disk
joblib.dump(model, 'age_prediction_model.joblib')

# Load the model from disk
model = joblib.load('age_prediction_model.joblib')

# Define a dictionary to map the numeric values back to the original names
label_to_name_dict = {label: name for name, label in zip([name[0] for name in X], X_encoded)}

# Define the Flask application
app = Flask(__name__)

# Define the route to handle user input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input for name
        name = request.form['name']

        # Get the numeric label for the input name
        label = le.transform([name])[0]

        if label not in X_encoded:
            error = 'Name not found in dataset.'
            return render_template('predict.html', error=error)
        else:
            # Use the model to predict the age
            age = model.predict([[label]])[0]
            name = label_to_name_dict[label]
            return render_template('predict.html', name=name, age=age)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
