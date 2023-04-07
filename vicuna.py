from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelWithLMHead

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-13b")
model = AutoModelWithLMHead.from_pretrained("eachadea/vicuna-13b")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['input_text']
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.8)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = {'output_text': output_text}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
