from flask import Flask, request,render_template
import pandas as pd
print(pd.__version__)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

train_dataset = pd.read_csv('data/train_data.csv')
X_train = train_dataset.iloc[:, 0:-1].values
y_train = train_dataset.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 5))
model.fit(X_train_scaled, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        data = {
            'Sex': request.form["Sex"],
            'GeneralHealth': request.form["GeneralHealth"],
            'PhysicalHealthDays': request.form["PhysicalHealthDays"],
            'MentalHealthDays': request.form["MeantalHealthDays"],
            'PhysicalActivities': request.form["PhysicalActivities"],
            'SleepHours': request.form["SleepHours"],
            'HadAngina': request.form["HadAngina"],
            'HadStroke': request.form["HadStroke"],
            'HadAsthma': request.form["HadAsthma"],
            'HadCOPD': request.form["HadCOPD"],
            'HadDepressiveDisorder': request.form["HadDepressiveDisorder"],
            'HadKidneyDisease': request.form["HadKidneyDisease"],
            'HadArthritis': request.form["HadArthritis"],
            'HadDiabetes': request.form["HadDiabetes"],
            'SmokeStatus': request.form["SmokeStatus"],
            'AgeCategory': request.form["AgeCategory"],
            'BMI': request.form["BMI"],
            'AlcoholDrinkers': request.form["AlcoholDrinkers"],
            'SleepCategory': request.form["SleepCategory"]
        }
        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability=model.predict_proba(input_scaled)

        if prediction[0] == 0:
            return render_template('prediction_0.html',probability=probability[0][0])
        else:
            return render_template('prediction_1.html',probability=probability[0][1])
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True, port=5001)
