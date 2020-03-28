from flask import Flask , render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close() 

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":

        MyDict=request.form
        fever = int(MyDict['fever'])
        Age = int(MyDict['Age'])
        BodyPain = int(MyDict['BodyPain'])
        RunnyNose = int(MyDict['RunnyNose'])
        DiffBreath = int(MyDict['DiffBreath'])

        inputFeatures = [fever, BodyPain, Age, RunnyNose, DiffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')

if __name__== "__main__":
    app.run(debug=True)