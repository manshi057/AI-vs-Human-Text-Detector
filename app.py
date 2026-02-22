from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# ✅ Load model once
clf_svm = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""   # ✅ Fix 1

    if request.method == 'POST':
        user_text = request.form['message']

        text = tfidf.transform([user_text])
        result = clf_svm.predict(text)

        if result[0] == 1:   # ✅ Fix 2
            message = 'The text is likely written by AI'
        else:
            message = 'The text is likely written by Human'

    return render_template('main.html', params=message)

if __name__ == '__main__':
    app.run(debug=True)
