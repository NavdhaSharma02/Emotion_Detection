from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probabilities = None
    user_text = ""
    if request.method == 'POST':
        user_text = request.form['user_text']
        sample_text = [user_text]
        prediction = pipe_lr.predict(sample_text)[0]
        probabilities = pipe_lr.predict_proba(sample_text)[0]
    return render_template('interface.html', prediction=prediction, probabilities=probabilities, user_text=user_text)

if __name__ == '__main__':
    app.run(debug=True)
    
    