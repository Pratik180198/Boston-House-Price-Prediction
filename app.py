from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('RFHouseModel.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predicting')
def add():
    return render_template("predicting.html")

@app.route("/predict", methods=['POST','GET'])
def predict():
    scaler = pickle.load(open('Scaler.pkl', 'rb'))
    l=[]
    if request.method == 'POST':
        CRIM = np.log(float(request.form['CRIM']))
        ZN = float(request.form['ZN'])
        INDUS = np.log(float(request.form['INDUS']))
        CHAS = request.form['CHAS']
        if (CHAS == 'Zero'):
            CHAS=0
        else:
            CHAS=1
        NOX = np.log(float(request.form['NOX']))
        RM = np.log(float(request.form['RM']))
        AGE = float(request.form['AGE'])
        DIS = np.log(float(request.form['DIS']))
        TAX = np.log(float(request.form['TAX']))
        PTRATIO = np.log(float(request.form['PTRATIO']))
        B = np.log(float(request.form['B']))
        LSTAT = np.log(float(request.form['LSTAT']))
        l.extend([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, TAX, PTRATIO, B, LSTAT])
        arr = np.asarray([l])
        scaler = scaler.transform(arr)
        output=1000*(round(model.predict(scaler)[0],2))
        if output < 0:
            return render_template('index.html', prediction_texts="Sorry you cannot sell this House")
        else:
            return render_template('predicting.html', prediction_text="You Can Sell this House at {} $".format(output))
    else:
        return render_template('index.html')





if __name__=="__main__":
    app.run(debug=True)