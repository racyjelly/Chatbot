from flask import Flask
# python web framework 
from flask import request
 # 웹 요청 관련 모듈
from flask import render_template, redirect, url_for, request
# flask에서 필요한 모듈
from flask import jsonify
# import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용
import processor


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']
        response = processor.predict(the_question)

    return jsonify({"response": response })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)