from flask import Flask,render_template,request
import pickle
import numpy as np
import datetime

app=Flask(__name__)
model_from_pkl=pickle.load(open('/home/salaryprediction/salary_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    # Format as [Position, Department, Age]
    sample = [int(request.values['position']),int(request.values['department']),int(request.values['age'])]
    np_sample = np.array(sample)
    np_sample = np.reshape(np_sample,(1,-1))
    l=model_from_pkl.predict(np_sample)
   
    # Get the result for display
    l=l.item()
    
    l='${:,.2f}'.format(l) # Showing salary in USD

    return render_template('result.html',predict=str(l))

if __name__=='__main__':
    app.run(port=8004)
