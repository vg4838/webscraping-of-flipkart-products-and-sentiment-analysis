import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def preprocess_data(dic_,prod_name):
    """
    Purpose: preprocess text (tokenize, stem, and remove stopwords)
    """

    # Open and read file
    # Read file

    # file = pd.DataFrame(dic_)
    # file.to_csv('sentiments.csv')
    # print("Reading file... \n")
    data = pd.read_csv("sentiments.csv", encoding="utf-8", index_col=False)

    # data = pd.read_csv('sentiments_res.csv')

    # Get the review text and title columns
    text_data = data['Comment'].to_numpy()
    title_data = data["CommentHead"].to_numpy()

    # Create English stop words list
    stop_words = set(stopwords.words('english'))

    # Instantiate a stemmer
    ps = PorterStemmer()

    # Instantiate a retokenizer
    detokenizer = TreebankWordDetokenizer()

    # Create a list for tokenized documents in loop
    tokenized_text = []
    res_text = []
    try:
        for i, text in enumerate(text_data):
            # Create lists
            pos, neg, neu = [], [], []

            # Split the text into sentences
            sentences = sent_tokenize(text)

            # Lower all the words and tokenize the sentences
            for sentence in sentences:
                sentence = sentence.lower()
                words = word_tokenize(sentence)

                # Remove stop words
                stopped_words = [w for w in words if not w in stop_words]

                # Stemmanize the words
                stem_words = [ps.stem(word) for word in stopped_words]

                # Retokenize the words into sentences
                new_sentence = detokenizer.detokenize(stem_words)

                # Calculate sentiment score
                neg_list, neu_list, pos_list, res = sentiment_score(new_sentence, pos, neg, neu)

            # Print the average sentiment score of each review
            text_fin = "Average negative score:"+ str(round(sum(neg) / max(1, len(neg)), 2))+"\n"
            text_fin+="Average neutral score:"+ str(round(sum(neu) / max(1,len(neu)),2))+"\n"
            text_fin+="Average positive score:"+ str(round(sum(pos) / max(1,len(pos)),2))

            dic_[i]['Sentiment'] = text_fin
            # print("Average negative score:", sum(neg) / max(1, len(neg)))
            # print("Average neutral score:", sum(neu) / max(1,len(neu)))
            # print("Average positive score:", sum(pos) / max(1,len(pos)))
            # print(
            #     "-------------------------------------------------------NEXT COMMENT-------------------------------------------------------------------")
    except Exception as e:
        print(e)
    return dic_

def sentiment_score(sentence, pos, neg, neu):
    # Create a SentimentIntensityAnalyzer object.
    sia_obj = SentimentIntensityAnalyzer()

    # Create a dictionary that contains positive, negative, neutral, and compound scores.
    sentiment_dict = sia_obj.polarity_scores(sentence)

    # Print the sentiment scores
    # print("Overall sentiment dictionary is: ", sentiment_dict)
    # print(sentence, "was rated as", sentiment_dict['neg'] * 100, "% Negative")
    # print(sentence, "was rated as", sentiment_dict['neu'] * 100, "% Neutral")
    # print(sentence, "was rated as", sentiment_dict['pos'] * 100, "% Positive")

    # Keep track of the sentiment score of each sentence in a review
    neg.append(sentiment_dict['neg'] * 100)
    neu.append(sentiment_dict['neu'] * 100)
    pos.append(sentiment_dict['pos'] * 100)

    res = "Sentence Overall Rated As"
    # print("Sentence Overall Rated As", end=" ")

    # Decide if the sentiment as positive, negative or neutral using the compound score
    if sentiment_dict['compound'] >= 0.05:
        # print("Positive")
        res += "Positive"

    elif sentiment_dict['compound'] <= - 0.05:
        # print("Negative")
        res += "Negative"

    else:
        # print("Neutral")
        res += "Neutral"

    # print("")

    # Return lists
    return (neg, neu, pos, res)

