#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
""" 
    ***
    
    Data (pre)processing.
    
    :authors: Elie MAZE, Luc Thomas  

"""


#---------------------------------------------------------------------- MODULES
import re
import spacy
from tqdm import tqdm
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt 
import seaborn as sns


#-------------------------------------------------------------------- FUNCTIONS
def findCAPSLOCK(comment):
    r = re.compile(r"[A-Z]")
    capslock = r.findall(comment)
    return len(capslock) / len(comment)

def find_chain_CAPSLOCK(comment):
    r = re.compile(r"[A-Z]{2,}")
    capslock = r.findall(comment)
    return len(capslock) / len(comment)

def find_exclamation(comment):
    r = re.compile(r"\!")
    exclamation = r.findall(comment)
    return len(exclamation) / len(comment)

def find_chain_exclamation(comment):
    r = re.compile(r"[\!]{2,}")
    exclamation = r.findall(comment)
    return len(exclamation) / len(comment)

def find_interogation(comment):
    r = re.compile(r"\?")
    interogation = r.findall(comment)
    return len(interogation) / len(comment)

def find_chain_interogation(comment):
    r = re.compile(r"[\?]{2,}")
    interogation = r.findall(comment)
    return len(interogation) / len(comment)

def find_etc(comment):
    r = re.compile(r"[\.]{2,}")
    etc = r.findall(comment)
    return len(etc) / len(comment)

def cleanReview(text):
    pattern_email = re.compile(r"[a-zA-Z0-9.-]+@[a-zA-Z.]+")
    pattern_url1 = re.compile(r"https?://[a-zA-Z0-9./]+")
    pattern_url2 = re.compile(r"www\.[a-zA-Z0-9.-:/]+")

    text = pattern_email.sub(" ", text)
    text = pattern_url1.sub(" ", text)
    text = pattern_url2.sub(" ", text)
    text = re.sub(r' {2,}', " ", text)
    return text

def raw_preprocess_word(token):
    if token.text.isnumeric(): return ""
    if token.is_stop: return ""
    if token.ent_type_ == "PUNCT": return ""
    word = re.sub(r'[\@\#\$\*\-\=\+\~\|\°]', " ", token.text)
    word = re.sub(r' {2,}', " ", word)
    return word.strip()

def check_tokens(df, n_jobs=8):
    nlp = spacy.load("fr_core_news_lg")
    gen = nlp.pipe(df.Commentaire.str.lower().map(cleanReview), disable=["tagger", "parser", "attribute_ruler", "textcat"], n_process=n_jobs, batch_size=100)

    stats_comments = []
    stats_words = []
    stats_nb_words = []
    for doc in tqdm(gen) : 
        words = [word for word in [raw_preprocess_word(token) for token in doc] if word]
        stats_comments.append(len(words))
        nb_words = [len(word) for word in words]
        if nb_words:
            stats_words += nb_words[:]
            stats_nb_words += [max(nb_words)]
        else:
            stats_nb_words += [0]

    df["nb_words"] = stats_comments
    df["words_max_len"] = stats_nb_words

    return stats_words

def preprocess_word(token):
    if token.text.isnumeric(): return ""
    if len(token.text)<2 or len(token.text)>20: return ""
    if token.is_stop: return ""
    if token.ent_type_ == "PUNCT": return ""
    word = re.sub(r'[\@\#\$\*\-\=\+\~\|\°]', " ", token.text)
    word = re.sub(r' {2,}', " ", word)
    return word.strip()

def preprocess_lemma(token):
    if token.text.isnumeric(): return ""
    if len(token.text)<2 or len(token.text)>20: return ""
    if token.is_stop: return ""
    if token.ent_type_ == "PUNCT": return ""
    return token.lemma_

def preprocess_comments(df, n_jobs=8):
    nlp = spacy.load("fr_core_news_lg")
    gen = nlp.pipe(df.Commentaire.str.lower().map(cleanReview), disable=["tagger", "parser", "attribute_ruler", "textcat"], n_process=n_jobs, batch_size=100)

    liste_lemma, list_words = [], []
    for doc in tqdm(gen) : 
        text_preprocess = " ".join(preprocess_word(token) for token in doc)
        list_words.append(text_preprocess)

        text_preprocess = " ".join(preprocess_lemma(token) for token in doc)
        liste_lemma.append(text_preprocess)

    df['cleaned_words'] = list_words
    df['cleaned_lemma'] = liste_lemma

def plot_word_cloud(text, title, stop_words, color="white", max_words=200, figsize=(12,6)) :

    # Définir le calque du nuage des mots
    wc = WordCloud(background_color=color, max_words=max_words, stopwords=stop_words, max_font_size=50, random_state=42)

    # Générer et afficher le nuage de mots
    plt.figure(figsize=figsize)
    plt.title(title)
    wc.generate(text)
    plt.imshow(wc)
    plt.show()

def return_ngram(texts, stop_words, ngram_range=(1,1)):
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    vectorizer.fit(texts)
    count_list = np.array(vectorizer.transform(texts).sum(0))[0]
    count_words = list(zip(vectorizer.get_feature_names_out(), count_list))
    count_words = sorted(count_words, key=lambda x: x[1], reverse=True)

    count_words = pd.DataFrame(count_words, columns=['word', 'count'])
    return count_words

def plotNgramsPerStars(df, stop_words, ngram=1, figsize=(10,6)):

    data = df.groupby('star').agg({'Commentaire':lambda x : ' '.join(x), 'star': 'count'})

    vectorizer = TfidfVectorizer( stop_words=stop_words, ngram_range=(ngram,ngram))
    vectorizer.fit(df.Commentaire)
    count_list = vectorizer.transform(data.Commentaire).toarray()

    count_words = np.concatenate([np.expand_dims(vectorizer.get_feature_names_out(),1), count_list.T], axis=1)

    count_words = pd.DataFrame(count_words, columns=['word', 'star1', 'star2', 'star3', 'star4', 'star5'])

    count_words[['star1', 'star2', 'star3', 'star4', 'star5']] = count_words[['star1', 'star2', 'star3', 'star4', 'star5']].astype(float)


    plt.figure(figsize=figsize)
    data_words = count_words.sort_values('star1', ascending=False).head(5)
    plt.subplot(121)
    sns.barplot(data=data_words, x='star1', y='word')
    plt.title('Star 1', fontsize=14)

    data_words = count_words.sort_values('star5', ascending=False).head(5)
    plt.subplot(122)
    sns.barplot(data=data_words, x='star5', y='word')
    plt.title('Star 5', fontsize=14)
    plt.show()