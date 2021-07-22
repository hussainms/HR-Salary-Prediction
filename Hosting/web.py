from flask import Flask,render_template,request
import pickle
import numpy as np
import datetime

app=Flask(__name__)
model_from_pkl=pickle.load(open('Hosting/salary_model.pkl','rb'))

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

    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        local_ip = request.environ['REMOTE_ADDR']
    else:
        local_ip = request.environ['HTTP_X_FORWARDED_FOR'] # if behind a proxy

    # write log
    file1 = open("Hosting/predication-log.csv","a") #append mode
    file1.write(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ',') #event time
    file1.write(local_ip + ',') #loging IP addresses
    file1.write(','.join(map(str, sample)) + ',') #input
    file1.write(str(l) + '\n') #prediction
    file1.close()

    l='${:,.2f}'.format(l) # Showing salary in USD

    return render_template('result.html',predict=str(l))

if __name__=='__main__':
    app.run(port=8004)
