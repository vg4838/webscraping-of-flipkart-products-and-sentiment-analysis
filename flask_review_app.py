# doing necessary imports

from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS,cross_origin
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq
import naive_bayes
import k_nearest_neighbors
import vader
import sys

app1 = Flask(__name__)  # initialising the flask app with the name 'app'


@app1.route('/', methods=['GET'])
def homepage():
    return render_template('index.html',lis = [])


# base url + /
# http://localhost:8000 + /
@app1.route('/scrap', methods=['POST'])  # route with allowed methods as POST and GET
def index():
    if request.method == 'POST':
        searchString = request.form['content'].replace(" ", "")  # obtaining the search string entered in the form
        try:
            flipkart_url = "https://www.flipkart.com/search?q=" + searchString  # preparing the URL to search the product on flipkart
            uClient = uReq(flipkart_url)  # requesting the webpage from the internet
            flipkartPage = uClient.read()  # reading the webpage
            uClient.close()  # closing the connection to the web server
            flipkart_html = bs(flipkartPage, "html.parser")  # parsing the webpage as HTML
            bigboxes = flipkart_html.findAll("div", {
                "class": "_2pi5LC col-12-12"})  # seacrhing for appropriate tag to redirect to the product link
            del bigboxes[
                0:3]  # the first 3 members of the list do not contain relevant information, hence deleting them.
            box = bigboxes[0]  # taking the first iteration (for demo)
            names = flipkart_html.findAll('div',  {"class": "_4rR01T"})
            # for name in names:
            #     print(name.text)
            productLink = "https://www.flipkart.com" + box.div.div.div.a['href']  # extracting the actual product link
            prodRes = requests.get(productLink)  # getting the product page from server
            prod_html = bs(prodRes.text, "html.parser")  # parsing the product page as
            prod_name = prod_html.findAll('h1', {"class": "yhB1nd"})[0].span.text
            rev = prod_html.findAll("div", {"class": "col JOpGWq"})
            all_rev_link = "https://www.flipkart.com" + rev[0].find_all('a')[-1]['href']
            all_rev_res = requests.get(all_rev_link)
            all_reviews_html = bs(all_rev_res.text, "html.parser")
            #             page wise
            commentboxes = all_reviews_html.find_all('div', {'class': "_1AtVbE col-12-12"})
            page = commentboxes[-1]
            reviews = []
            loop = (page.div.div.span.text).split(" ")[3]
            loop = loop.replace(",","")
            for i in range(1,int(loop)):
                for commentbox in commentboxes[4:-1]:
                    try:
                        # _16PBlm
                        # name = commentbox.div.div.find_all('p', {'class': '_3LYOAd _3sxSiS'})[0].text
                        name = commentbox.div.div.find_all('p', {'class': '_2sc7ZR _2V5EHH'})[0].text

                    except:
                        name = 'No Name'

                    try:
                        rating = commentbox.div.div.div.div.div.text

                    except:
                        rating = 'No Rating'

                    try:
                        commentHead = commentbox.div.div.div.p.text
                    except:
                        commentHead = 'No Comment Heading'
                    try:
                        comtag = commentbox.div.div.find_all('div', {'class': ''})
                        custComment = comtag[0].div.text
                    except:
                        custComment = 'No Customer Comment'
                    mydict = {"Product": prod_name, "Name": name, "Rating": rating, "CommentHead": commentHead,
                              "Comment": custComment}  # saving that detail to a dictionary
                    reviews.append(mydict)  # appending the comments to the review list

                page_link = "https://www.flipkart.com" + page.find_all('a')[-1]['href'][:-1]+str(i+1)
                page_res = requests.get(page_link)
                page_html = bs(page_res.text, "html.parser")
                commentboxes = page_html.find_all('div', {'class': "_1AtVbE col-12-12"})
                if i == 8:
                    break
            reviews_knn, train, test, report = k_nearest_neighbors.run_k_nearest_neighbors(reviews)
            reviews_vader = vader.preprocess_data(reviews, prod_name)
            # reviews_naive, acc, fea = naive_bayes.naive_bayes(reviews)
            r = list(zip(reviews_knn, reviews_vader))
            return render_template('results.html', reviews= r,train=train, test = test, report = report)  # showing the review to the user
        except Exception as e:
            print(e)
            return 'something is wrong'


if __name__ == "__main__":
    app1.run(port=8001, debug=True)  # running the app on the local machine on port 8000
