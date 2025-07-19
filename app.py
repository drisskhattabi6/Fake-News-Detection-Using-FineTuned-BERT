from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Define BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load('model/fakenews_weights.pt', map_location=torch.device('cpu')))
model.eval()

# Init Flask
app = Flask(__name__)

def predict_fake_news(text):
    tokens = tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt"
    )
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output, dim=1).item()

    return "Fake News" if prediction == 1 else "Real News"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    prediction = predict_fake_news(text)
    return render_template('result.html', news=text, prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400
    result = predict_fake_news(data['text'])
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
