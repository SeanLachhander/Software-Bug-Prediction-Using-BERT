import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import BugPredictor
import functools
import tokenizers
from tokenizers import BertWordPieceTokenizer
import torch.nn as nn


app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()
tokenizer = config.TOKENIZER
device = "cpu"


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


def predict_git_category(sample_message):
    encoded_message = tokenizer.encode_plus(sample_message, max_length=512, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded_message['input_ids'].to(device)
    attention_mask = encoded_message['attention_mask'].to(device)
    
    output = bug_predictor_model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction_idx = torch.max(output, dim=1)
    features = ['Bug', 'Feature', 'Question']

    return features[class_names[prediction_idx]]


@app.route("/predict")
def predict():
    req_data = request.get_json()
    sentence = req_data['sample_message']
    start_time = time.time()
    prediction = predict_git_category(sentence)
    response = {}
    response["response"] = {
        "prediction": prediction,
        "sample_message": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    class_names = [0, 1, 2]
    global bug_predictor_model 
    bug_predictor_model = BugPredictor(len(class_names))
    bug_predictor_model.load_state_dict(torch.load("best_model_2.bin"))
    bug_predictor_model.to("cpu")
    bug_predictor_model.eval()
    app.run(host="0.0.0.0", port="9999")