import json
import plotly
import pandas as pd
from plotly.graph_objs import Bar, Layout, Figure, Histogram

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    top_3_categories = df.iloc[:, 4:].sum().sort_values(ascending=False)[:3]/len(df)
    top_3_cat_names = top_3_categories.index
    
    hist_num_cats = df.iloc[:, 4:].sum(axis=1)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = []
    
    graph_one = []
    graph_one.append(
          Bar(
          x = genre_names,
          y = genre_counts,
          )
      )

    layout_one = Layout(title = 'Distribution of Message Genres',
                xaxis = {'title': 'Count'},
                yaxis = {'title':'Genre'},
                )
    
    graph_two = []
    graph_two.append(
          Bar(
          x = top_3_cat_names,
          y = top_3_categories,
          )
      )

    layout_two = Layout(title = 'Top 3 most occuring Categories',
                xaxis = {'title': 'Categories'},
                yaxis = {'title':'Percentage'},
                )
    
    graph_three = []
    graph_three.append(
          Histogram(
          x = hist_num_cats,
          )
      )

    layout_three = Layout(title = 'Distribution of # of Categories per Message',
                xaxis = {'title': '# of Categories'},
                yaxis = {'title':'Counter'},
                )
    
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()