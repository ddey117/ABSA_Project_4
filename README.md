## Twitter Sentiment Classifier
### Gaining actionable insight from social media data

Author: Dylan Dey

This project it available on github here: link

The Author can be reached at the following email: ddey2985@gmail.com

associated blog for BERT transfer learning using a sklearn wrapper to easily create a powerful sentiment classifier with a small dataset can be found at link below.

[Blog Link](https://dev.to/ddey117/quick-bert-pre-trained-model-for-sentiment-analysis-with-scikit-wrapper-3jcp)

## Overview
Process twitter text data to gain insights on a brand and associated products. Create a machine learning sentiment classifier in order to predict sentiment in never before seen tweets. Create word frequency distributions, wordclouds, bigrams, and quadgrams to easily assess actionable insight to address concerns for the brand and it's product line.

## Business Problem
A growing company with an established social media presence wants to explore options for generating actionable insights from twitter text data in a more efficient way. They have a new product releasing this year and are interested in what their customers feel about their products.

The company wants a proof of concept for a machine learning solution to this problem. Why would it be worth the time and resources? How can you easily gain actionable insight from a large collection of tweets? Can we trust the model to make accurate predictions?

## The Data
Apple hosted an SXSW event in 2011 that took advantage of their release party to crowdsource some data labeling and boost their social media traffic for the event.

Using this data, sourced from CrowdFlower, as well as some data from an additional Apple Twitter Sentiment Dataset also made available from CrowdFlower and data.world but cleaned and processed and made available on kaggle by author Chanran Kim, a machine learning classifier will be created in order to predict for sentiment contained within a tweet and show how it could be used in tandem with some NLP techniques to extract actionable insights from cluttered tweet data in a manageable way.

## Function Definition

All functions used to preprocess twitter data, such as removing noise from text and tokenizing, as well as the functions for creating confusion plots to quickly assess performance are shown below.

```
#list of all functions for modeling
#and processing

#force lowercase of text data
def lower_case_text(text_series):
    text_series = text_series.apply(lambda x: str.lower(x))
    return text_series

#remove URL links from text
def strip_links(text):
    link_regex = re.compile('((https?):((\/\/)|(\\\\))+([\w\d:#@%\/;$()~_?\+-=\\\.&](#!)?)*)|{link}/gm')
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

#remove '@' and '#' symbols from text
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

#tokenize text and remove stopwords
def process_text(text):
    tokenizer = TweetTokenizer()
    
    stopwords_list = stopwords.words('english') + list(string.punctuation)
    stopwords_list += ["''", '""', '...', '``']
    my_stop = ["#sxsw",
               "sxsw",
               "sxswi",
               "#sxswi's",
               "#sxswi",
               "southbysouthwest",
               "rt",
               "tweet",
               "tweet's",
               "twitter",
               "austin",
               "#austin",
               "link",
               "1/2",
               "southby",
               "south",
               "texas",
               "@mention",
               "ï",
               "ï",
               "½ï",
               "¿",
               "½",
               "link", 
               "via", 
               "mention",
               "quot",
               "amp",
               "austin"
              ]

    stopwords_list +=  my_stop 
    
    tokens = tokenizer.tokenize(text)
    stopwords_removed = [token for token in tokens if token not in stopwords_list]
    return stopwords_removed
    


#master preprocessing function
def Master_Pre_Vectorization(text_series):
    text_series = lower_case_text(text_series)
    text_series = text_series.apply(strip_links).apply(strip_all_entities)
    text_series = text_series.apply(unidecode.unidecode).apply(html.unescape)
    text_series =text_series.apply(process_text)
    lemmatizer = WordNetLemmatizer()
    text_series = text_series.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return text_series.str.join(' ').copy()


#function for intepreting results of models
#takes in a pipeline and training data
#and prints cross_validation scores 
#and average of scores


def cross_validation(pipeline, X_train, y_train):
    scores = cross_val_score(pipeline, X_train, y_train)
    agg_score = np.mean(scores)
    print(f'{pipeline.steps[1][1]}: Average cross validation score is {agg_score}.')


#function to fit pipeline
#and return subplots 
#that show normalized and 
#regular confusion matrices
#to easily intepret results
def plot_confusion_matrices(pipe):
    
    pipe.fit(X_train, y_train)
    y_true = y_test
    y_pred = pipe.predict(X_test)

    matrix_norm = confusion_matrix(y_true, y_pred, normalize='true') 
    matrix = confusion_matrix(y_true, y_pred) 

    fig, (ax1, ax2) = plt.subplots(ncols = 2,figsize=(10, 5))
    sns.heatmap(matrix_norm,
                annot=True, 
                fmt='.2%', 
                cmap='YlGn',
                xticklabels=['Pos_predicted', 'Neg_predicted'],
                yticklabels=['Positive Tweet', 'Negative_Tweet'],
                ax=ax1)
    sns.heatmap(matrix,
                annot=True, 
                cmap='YlGn',
                fmt='d',
                xticklabels=['Pos_predicted', 'Neg_predicted'],
                yticklabels=['Positive Tweet', 'Negative_Tweet'],
                ax=ax2)
    plt.show();

    
#loads a fitted model from memory 
#returns confusion matrix and
#returns normalized confusion matrix
#calculated using given test data
def confusion_matrix_bert_plots(model_path, X_test, y_test):
    
    model = load_model(model_path)
    
    y_pred = model.predict(X_test)

    matrix_norm = confusion_matrix(y_test, y_pred, normalize='true')

    matrix = confusion_matrix(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(ncols = 2,figsize=(10, 5))
    sns.heatmap(matrix_norm,
                annot=True, 
                fmt='.2%', 
                cmap='YlGn',
                xticklabels=['Pos_predicted', 'Neg_predicted'],
                yticklabels=['Positive Tweet', 'Negative_Tweet'],
                ax=ax1)
    sns.heatmap(matrix,
                annot=True, 
                cmap='YlGn',
                fmt='d',
                xticklabels=['Pos_predicted', 'Neg_predicted'],
                yticklabels=['Positive Tweet', 'Negative_Tweet'],
                ax=ax2)
    plt.show();
```


### Class Imbalance of Dataset

The twitter data used for this project was collected from multiple sources from [CrowdFlower](https://appen.com/datasets-resource-center/). The project will only focus on binary sentiment (positive or negative). The total amount of tweets and associated class balances are show below. This distribution is further broken down by brand in the chart below the graphs.

![Class_Imbalance_Image](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/pvpg3napakuqebqq4j3p.jpg)


#### Apple Positive vs Negative Tweet Counts
positive    0.654194
negative    0.345806
+++++++++++++++++++++++
positive    2028
negative    1072
+++++++++++++++++++++++++++++++++++++++++++++++++++
#### Google Positive vs Negative Tweet Counts
positive    740
negative    136
+++++++++++++++++++++++
positive    0.844749
negative    0.155251

### Data Exploration By Brand

Below are some quick examinations of the distrubtion of tweets by brand and sentiment.

Apple Positive vs Negative Tweet Counts
positive    0.654194
negative    0.345806

positive    2028
negative    1072

++++++++++++++++++++++++++++++++++++++++

Google Positive vs Negative Tweet Counts
positive    740
negative    136

positive    0.844749
negative    0.155251




After being seperated by brand/emotion pairs, the twitter text data will be processed and cleaned. The text data will then be tokenized using a scikit learn Twitter tokenizer before creating term frequency counts for each brand/emotion combination using the tokenized text data. The term frequency counts are used to generate word clouds to quickly visualize what people do and do not like about the brand or product. Bigrams, trigrams, and quadrgrams were created using [Pointwise Mutual Information(PMI)](https://en.wikipedia.org/wiki/Pointwise_mutual_information) scores generated using NLTK collocations.

Pointwise mutual information can be used to determine if two words co-occur by chance or have a high probability to express a unique concept. This concept can be expanded to determine if three words have a high probability to occur together, four and so on.

These Natural Language Processing (NLP) techniques and others can easily be used to make actionable insight from twitter data.




## Word Clouds
### Positive Apple Tweets
![pos_appl_cloud](images/pos_apple_cloud.jpg)

### Negative Apple Tweets
![neg_appl_cloud](images/neg_apple_cloud.jpg)

### Positive Google Tweets
![pos_google_cloud](images/pos_google_cloud.jpg)

### Negative Google Tweets
![neg_google_cloud](images/neg_google_cloud.jpg)

###### Please See Notebook for bigrams, trigrams, quadrgrams.



#### Some observations from exploring the data:

- Multiple complaints about issues with iphone 6 and its new touch id feature. Some googling unveiled an issue in which iphone 6 touch id button / home button would malfunction and heat up to high temperatures. 
- many complaints about phone chargers 
- high negative sentiment for iphone batteries 
- Some users displeased with issues with apple news app
- apple ipad 2 described as a design headache
- Complaints about customer service
- public image described as fascist 


Recommend to focus on improving battery life and quality. Improve phone accessories for charging and protecting batteries. (apple did improved a lot on this since 2011 when many of the tweets were collected)

Address technical issues with iphone 6 and apple news app crashing.

Launch a public relations campaign and give back to the community to boost public image.

Reassess training protocols for customer facing employees and ensure customer service is a cornerstone of Apple culture.



#### Proof of Concept
Actionable insight can be gained with enough social media data. A reasonable amount of labeled data can be budgeted for a growing business in order to train a machine learning sentiment classifier on that data and deploy it in order to gain more insights into consumer sentiment on your brand or products. 


## Data Modeling

#### Classification Metric Understanding
![Confusion_Matrix_Breakdown](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/1sq4f1wehvjntw5lwbzt.jpg)

#### Confusion Matrix Description

# Apple Tweet Sentiment Analysis
## Modeling Notebook

Author: Dylan Dey

The Author can be reached at the following email: ddey2985@gmail.com

Blog: [Quick BERT Pre-Trained Model for Sentiment Analysis with Scikit Wrapper](https://dev.to/ddey117/quick-bert-pre-trained-model-for-sentiment-analysis-with-scikit-wrapper-3jcp)

#### Classification Metric Understanding
![Matrix_Understanding](images/Apple_Twitter_matrix_explained.jpg)

#### Confusion Matrix Description

There will always be some error involved in creating a predictive model. The model will incorrectly identify positive tweets as negative and vice versa. That means the error in any classification model in this context can be described by ratios of true positives or negatives vs false positives or negatives.


Correctly predicting a tweet to have negative sentiment is at the heart of the model, as this is the situation in which a company would have a call to action. An appropriately identified tweet with negative sentiment can be properly examined using some simple NLP techniques to get a quick buy effective way to view what is upsetting customers about the company it's products.

Correctly predicting a tweet to have positive sentiment is also important. Word frequency analysis can be used to summarize what consumers think Apple is doing right and also what consumers like about Apple's competitors. 

A false positive would occur when the model incorrectly identifies a tweet containing negative sentiment as a tweet that contains positive sentiment. Given the context of the business model, this would mean more truly negative sentiment will be left out of analyzing key word pairs for negative tweets. This could be interpreted as loss in analytical ability for what we care about most given the buisness problem: making informed decisions from information directly from consumers in the form of social media text. Minimizing false positives is important.

False negatives are also important to consider. A false negative would occur when the model incorrectly identifies a tweet that contains positive sentiment as one that contains negative sentiment. Given the context of the business problem, this would mean extra noise added to the data when trying to isolate for negative sentiment of brand/product. 

In summary, overall accuracy of the model and a reduction of both false negatives and false positives are the most important metrics to consider when developing the sentiment analyisis model. 

##### MVP Metric
[balanced_accuracy_score](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

From the documentation: 

"The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class."

This is a great metric for this problem as optimizing for the average of recall for each class will give the best performance given the context of the buisness problem. 


For comparison, I trained four different supervised learning classifiers using term frequency–inverse document frequency(TF-IDF) vectorized preprocessed tweet data. While the vectorization will not be needed for the BERT classifier, it is needed for these supervised classifiers. The best model was then be fitted with a random grid-searchCV to tune hyperparamters for best balanced accuracy. 

[TF-IDF wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

[TfidfVectorize sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

[MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)

[Random Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[Balanced Random Forest Classifier Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)

[XGBoosted Trees Documentation](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)


### Multinomial Naive Bayes Base Model Performance
![NB_Matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/w2ntksg37392a4ch22u4.png)

### Random Forest Classifier Base Model Performance
![RF_Matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/98t4fd8tg87z7amk3oh8.png)

### Balanced Random Forest Classifier Base Model Performance
![Balanced_RF_Matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/tec244erxdn4s5rf059h.png)

### XGBoosted Random Forest Classifier Base Model Performance
![XGBoosted_Matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/bamwgzjotqhdess6ej0h.png)

#### Code Block for randomized searchCV
```
#initialize grid search variables
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
criterion = ["gini", "entropy"]
min_samples_split = [8, 10, 12]
max_depth = [int(x) for x in np.linspace(10, 1000, num = 10)] 
min_samples_leaf = [0.01, 0.1, 1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

#rrandomly iterate 1667*3 times through the grid
balanced_rfc_rs = RandomizedSearchCV(estimator = BalancedRandomForestClassifier(), 
                                     param_distributions = random_grid,
                                     scoring = 'balanced_accuracy',
                                     n_iter = 1667,
                                     cv = 3,
                                     verbose=2,
                                     random_state=11,
                                     n_jobs = -1
                                    )


#fit random grid search and determine best_estimator_
balanced_rfc_rs.fit(tf_idf_X_train, y_train)

#create pipeline for best result from random grid search
balanced_rfc_rs_pipe = make_pipeline(vectorizer, 
                                     balanced_rfc_rs.best_estimator_)
```

#### See Below for resluts of random searchCV
![Best_supervised_performance](images/best_balanced_rf_matrix.jpg)


### Transfer Learning: Taking Advantage of a pre-trained BERT_base Classifier

Now that the supervised learning models have been built, trained, and tuned without any pre-training, our focus will now turn to transfer learning using Bidirectional Encoder Representations from Transformers(BERT), developed by Google. BERT is a transformer-based machine learning technique for natural language processing pre-training. BERTBASE models are pre-trained from unlabeled data extracted from the BooksCorpus with 800M words and English Wikipedia with 2,500M words. 

[Click Here for more from Wikipedia](https://en.wikipedia.org/wiki/BERT_(language_model))

[GitHub for BERT release code](https://github.com/google-research/bert)

Sckit-learn wrapper provided by Charles Nainan. [GitHub of Scikit Learn BERT wrapper](https://github.com/charles9n/bert-sklearn). 

This scikit-learn wrapper is used to finetune Google's BERT model and is built on the huggingface pytorch port.

The BERT classifier is now ready to be fit and trained on data in the same way you would any sklearn model. 

See the code block below for a quick example.
```
bert_1 = BertClassifier(do_lower_case=True,
                        train_batch_size=32,
                        max_seq_length=50
                       )

bert_1.fit(X_train, y_train)

y_pred = bert_1.predict(X_test)

```

Four models were trained and stored locally. See the code block below for the chosen parameters in every model.

```
"""
The first model was fitted as seen commeted out below 
after some trial and error to determine an appropriate
max_seq_length given my computer's capibilities. 

"""


# bert_1 = BertClassifier(do_lower_case=True,
#                       train_batch_size=32,
#                       max_seq_length=50
#                      )



"""
My second model contains 2 hidden layers with 600 neurons. 
It only passes over the corpus one time when learning.
It trains fast and gives impressive results.

"""


# bert_2 = BertClassifier(do_lower_case=True,
#                       train_batch_size=32,
#                       max_seq_length=50,
#                       num_mlp_hiddens=500,
#                       num_mlp_layers=2,
#                       epochs=1
#                      )

"""
My third bert model has 600 neurons still but
only one hidden layer. However, the model
passes over the corpus 4 times in total
while learning.

"""

# bert_3 = BertClassifier(do_lower_case=True,
#                       train_batch_size=32,
#                       max_seq_length=50,
#                       num_mlp_hiddens=600,
#                       num_mlp_layers=1,
#                       epochs=4
#                      )

"""
My fourth bert model has 750 neurons and 
two hidden layers. The corpus also gets
transversed four times in total while 
learning.

"""

# bert_4 = BertClassifier(do_lower_case=True,
#                       train_batch_size=32,
#                       max_seq_length=50,
#                       num_mlp_hiddens=750,
#                       num_mlp_layers=2,
#                       epochs=4
#                      )
```


#### Bert 1 Results
![Bert1_matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/bk45s73uuu3g3uq32gh6.jpg)

#### Bert 2 Results
![Bert2_matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zkf97e7mq70s0t3cn6ae.jpg)

#### Bert 3 Results
![Bert3_matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/9h1fzlqeo4hzrc0u2odq.jpg)

#### Bert 4 Results
![Bert4_matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/bhmnbot0ba1m2t4yt7ga.jpg)


## Evaluation

The best performing model was the  BERT Classifier with 4 epochs, one hidden layer, and 600 neurons. This classifier was able to correctly predict over 80% of negative tweets correctly, which is really impressive given the imbalance in the original data. It also correctly identifies positive tweets nearly 94% of the time.


##### Balanced Random Forest Confusion Matrix
![Bert3_Matrix](images/bert3_matrix.jpg)


While the BERT classifier performed the best, the balanced random forest classifier has moderate predictive abilities using sparse vectors.   

##### Balanced Random Forest Confusion Matrix
![Balanced_RandomForest_Matrix](images/best_balanced_rf_matrix.jpg)


## Conclusions 

- Either classifier could be used to predict sentiment on new brand-centric social media data for the company's own products or that of a competitor.

### Future Work

- Use the BERT classifier to predict the sentiment on new unlabeled twitter data filtered for product or brand of interest (Apple/Google) from another source to find more actionable insights to further proof of concept.


- Use the BERT classifier to predict the sentiment on new twitter data to help balance existing dataset and retrain the other models.

- leverage a state-of-the-art early stopping algorithm (ASHA) using Ray Tune and PyTorch.(1)(2)

(1)Author Amog Kamsetty explores the importance of hyperparameter tuning in his blog [Hyperparameter Optimization for Transformers: A guide
](https://medium.com/distributed-computing-with-ray/hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b). [This Colobrative Notebook](https://colab.research.google.com/drive/1tQgAKgcKQzheoh503OzhS4N9NtfFgmjF?usp=sharing) shared in the blog is a good starting point to try optimize with Ray Tune. 

(2)Author Richard Liaw shares a blog that shows how simple it is to leverage all of the cores and GPUs on your machine to perform parallel asynchronous hyperparameter tuning and how to launch a massive distributed hyperparameter search on the cloud (and automatically shut down hardware after completion). [Ray Tune: a Python library for fast hyperparameter tuning at any scale](https://towardsdatascience.com/fast-hyperparameter-tuning-at-scale-d428223b081c) also showcases a lot of exciting algorithms to explore when tuning models. 



## For More Information

Please review our full analysis in the [Exploratory Jupyter Notebook](./Apple_Twitter_Sentiment_Exploratory_Notebook.ipynb) and the [Modeling Jupyter Notebook](Apple_Twitter_Sentiment_Modeling.ipynb) or the [presentation](./Project_Presentation.pdf).

For any additional questions, please contact:

Author Name: Dylan Dey

Email: ddey2985@gmail.com

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── README.md                     <- The top-level README for reviewers of this project
├── Exploratory_Notebook.ipynb    <- exploratory notebook
├── Exploratory_Notebook.pdf      <- PDF version of exploratory notebook
├── Sentiment_Modeling.ipynb      <- modeling notebook
├── Sentiment_Modeling.pdf        <- modeling notebook pdf
├── Project_Presentation.pdf      <- project presentation pdf
├── data                          <- Both sourced externally and generated from code
└── images                        <- Both sourced externally and generated from code
```
