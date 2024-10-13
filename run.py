import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model from the CardiffNLP RoBERTa model
MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)  # Move the model to GPU
MAX_TOKENS = 512  # Maximum tokens for the model


def polarity_scores_roberta(text):
    """
    Calculates sentiment scores for the given text using RoBERTa model.

    The function takes a text input, tokenizes it using the RoBERTa tokenizer,
    feeds it to the pre-trained sentiment model, and calculates softmax probabilities
    to determine the polarity (Negative, Neutral, Positive) of the text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the sentiment scores:
            - 'Negative': Probability of negative sentiment.
            - 'Neutral': Probability of neutral sentiment.
            - 'Positive': Probability of positive sentiment.
    """
    encoded_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors='pt'
    ).to(device)  # Move the tensor to GPU
    with torch.no_grad():  # Disable gradient computation
        output = model(**encoded_text)
    scores = output[0][0].cpu().numpy()  # Move the scores back to CPU before converting to NumPy
    scores = softmax(scores)
    scores_dict = {
        'Negative': scores[0],
        'Neutral': scores[1],
        'Positive': scores[2]
    }
    return scores_dict


def sentiment_analysis(jsonData, textKey):
    """
    Perform sentiment analysis on a dataset of text entries.

    Iterates through the JSON data, extracts text, calculates polarity scores,
    and aggregates sentiment information.

    Args:
        jsonData (list): A list of dictionaries containing the text data for analysis.
        textKey (str): The key used to extract text from each row in the dataset.

    Returns:
        tuple: A tuple containing:
            - new_df (pd.DataFrame): A DataFrame with sentiment scores for each entry.
            - all_text (list): A list of all text entries analyzed.
            - total_reviews (int): Total number of reviews in the dataset.
            - total_reviews_analyzed (int): Number of reviews successfully analyzed.
            - number_of_negative (int): Count of negative reviews.
            - number_of_positive (int): Count of positive reviews.
            - number_of_neutral (int): Count of neutral reviews.
    """
    df = pd.DataFrame(jsonData)
    total_reviews = len(df)
    all_text = []
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row[textKey]
            all_text.append(text)
            res[i] = polarity_scores_roberta(text)
        except RuntimeError:
            print(f'Broke for id {i}, because the size of the text is too big for the model to process ')    

    new_df = pd.DataFrame(res).T
    total_reviews_analyzed = len(new_df)

    number_of_positive = ((new_df['Positive'] > new_df['Negative']) & (new_df['Positive'] > new_df['Neutral'])).sum()
    number_of_neutral = ((new_df['Neutral'] > new_df['Positive']) & (new_df['Neutral'] > new_df['Negative'])).sum()
    number_of_negative = ((new_df['Negative'] > new_df['Positive']) & (new_df['Negative'] > new_df['Neutral'])).sum()

    return new_df, all_text, total_reviews, total_reviews_analyzed, number_of_negative, number_of_positive, number_of_neutral


app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/', methods=['POST'])
def send_json():
    """
    Flask route to handle POST requests and perform sentiment analysis.

    The function receives JSON data, performs sentiment analysis on the
    text entries, and returns a JSON response with aggregated sentiment results.

    Returns:
        flask.Response: A JSON response with sentiment data for each entry in the request.
    """
    jsonData = request.get_json()
    allResponse = []

    for data in jsonData:       
        new_df, all_text, total_reviews, total_reviews_analyzed, number_of_negative, number_of_positive, number_of_neutral = sentiment_analysis(data['json'], data['message'])
        all_negative = new_df["Negative"].tolist()
        all_positive = new_df["Positive"].tolist()
        all_neutral = new_df["Neutral"].tolist()

        response = {
            "name": data['name'],
            "uniqueID": data['uniqueID'],
            "total_reviews": int(total_reviews),
            "total_reviews_analyzed": int(total_reviews_analyzed),
            "number_of_negative": int(number_of_negative),
            "number_of_positive": int(number_of_positive),
            "number_of_neutral": int(number_of_neutral),
            "all_negative": all_negative,
            "all_positive": all_positive,
            "all_neutral": all_neutral,
            "all_text": all_text
        }
        allResponse.append(response)

    return jsonify(allResponse)


if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000)
