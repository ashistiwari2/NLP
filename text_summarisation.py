#genism package
from gensim.summarization.summarizer import summarize
# NLTK Packages
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import nltk
import re
nltk.download('stopwords')
import streamlit as st
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#SPACY Packages
#-------------------------------------------------
#sumy 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
def sumy(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer=TextRankSummarizer()
    summary=summarizer(parser.document,2)
    text_summary=""
    for sentence in summary:
        text_summary+=str(sentence)
    return text_summary

#------------------------
#---------------------
#LEX RANK
from sumy.summarizers.lex_rank import LexRankSummarizer
def lex_rank(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, 2)
    dp = []
    for i in summary:
        lp = str(i)
        dp.append(lp)
    final_sentence = ' '.join(dp)
    return final_sentence
#-----------------------------------
#--------------------------------
#using Lunh
from sumy.summarizers.luhn import LuhnSummarizer
def luhn(docx):
    
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer_luhn = LuhnSummarizer()
    summary_1 = summarizer_luhn(parser.document, 2)
    dp = []
    for i in summary_1:
        
        lp = str(i)
        dp.append(lp)
    final_sentence = ' '.join(dp)
    return final_sentence
#--------------------------------------
#---------------------------------------
#LSA Latent Semantic Analyzer (LSA)
from sumy.summarizers.lsa import LsaSummarizer
def lsa(docx):
    
    
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer_lsa = LsaSummarizer()
    summary_2 = summarizer_lsa(parser.document, 2)
    dp = []
    for i in summary_2:
        lp = str(i)
        dp.append(lp)
    final_sentence = ' '.join(dp)
    return final_sentence
#----------------------------------------

#Function for NLTK
def _create_frequency_table(text_string) -> dict:
    print("------------------------------")
    print("frequency func executed well\n")
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable
    sent_tokenize(text_string)
def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()
    print("------------------------------")
    print("score func executed well\n")
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue
def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    print("------------------------------")
    print("average func executed well\n")
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average
def _generate_summary(sentences, sentenceValue, threshold):
    print("------------------------------")
    print("generate summary fun executed well\n")
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def nltk_summarizer(text):
    print("------------------------------")
    print("nltk func executed well\n")
    freq_table=_create_frequency_table(text)
    sentences=sent_tokenize(text)
    sentence_score=_score_sentences(sentences,freq_table)
    threshold=_find_average_score(sentence_score)
    summary=_generate_summary(sentences,sentence_score,1.5*threshold)
    return summary

#Function for SPACY
def spacy_summarizer(docx):
    print("------------------------------")
    print("spacy summ func executed well\n")
    # stopwords=list(STOP_WORDS) #buiding a list of stopword
    nlp=spacy.blank("en")
    #nlp=spacy.load('en')
    nlp.add_pipe('sentencizer')
    docx=nlp(docx)
    # len(list(docx.sents))
    # keyword=[]
    # stopwords=list(STOP_WORDS)
    # pos_tag=['PROPN','ADJ','NOUN','VERB']
    # for token in docx:
    #     if(token.text in stopwords or token.text in punctuation):
    #         continue
    #     if(token.pos_ in pos_tag):
    #         keyword.append(token.text)
    # freq_word=Counter(keyword)
    # freq_word.most_common(7)
    # max_freq=Counter(keyword).most_common(1)[0][1]
    # for word in freq_word.keys():
    #     freq_word[word]=(freq_word[word]/max_freq)
    # freq_word.most_common(5)
    # sent_strength={}
    # for sent in docx.sents:
    #     for word in sent:
    #         if word.text in freq_word.keys():
    #             if sent in sent_strength.keys():
    #                 sent_strength[sent]+=freq_word[word.text]
    #             else:
    #                 sent_strength[sent]=freq_word[word.text]
    # print(sent_strength)
    # summarised_sentences=nlargest(3,sent_strength,key=sent_strength.get)
    # print(summarised_sentences)
    # print(type(summarised_sentences[0]))
    # final_sentences=[w.text for w in summarised_sentences]
    # summary=' '.join(final_sentences)
    # print(summary)
    # return summary

    # mytoken=[token.text for token in docx1]
    #build word frequency
    word_frequencies={}
    for word in docx:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text]=1
        else:
            word_frequencies[word.text]+=1
    #print(word_frequencies)
    #maximum wword frequencies
    maximum_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=(word_frequencies[word]/maximum_frequency)
    #frequency table
    #print(word_frequencies)
    #sentence tokens
    sentence_list=[sentence for sentence in docx.sents]
    print(sentence_list)
    #sentence score
    sentence_scores={}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' '))<30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent]=word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent]+=word_frequencies[word.text.lower()]

    summarised_sentences=nlargest(3,sentence_scores,key=sentence_scores.get)
    print(summarised_sentences)
    for w in summarised_sentences:
        print(w.text)
    final_sentences = [w.text for w in summarised_sentences]
    summary = ''.join(final_sentences)

    return summary
    print(summary)

st.title("Text Summarizer App")
activities = ["Summarize Via Text"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == 'Summarize Via Text':
    st.subheader("Summary using NLP")
    article_text = st.text_area("Enter Text Here", "Type here")
    # cleaning of input text
    article_text = re.sub(r'\\[[0-9]*\\]', ' ', article_text)
    article_text = re.sub('[^a-zA-Z.,]', ' ', article_text)
    article_text = re.sub(r"\b[a-zA-Z]\b", '', article_text)
    article_text = re.sub("[A-Z]\Z", '', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    summary_choice = st.selectbox("Summary Choice", ["NLTK", "SPACY","SUMY","Genism","LEX Rank","Luhn","LSA"])
                                                     # "Genism"])
    if st.button("Summarize Via Text"):
        if summary_choice == 'NLTK':
            summary_result = nltk_summarizer(article_text)

            print(summary_result)
        elif summary_choice == 'SPACY':
            summary_result = spacy_summarizer(article_text)
        elif summary_choice == 'SUMY':
            summary_result = sumy(article_text)

            print(summary_result)
        elif summary_choice == 'Genism':
            summary_result = summarize(article_text)
        elif summary_choice == 'LEX Rank':
            summary_result = lex_rank(article_text)
        elif summary_choice == 'Luhn':
            summary_result = luhn(article_text)
        elif summary_choice == 'LSA':
            summary_result = lsa(article_text)
        print(summary_result)
        st.write(summary_result)

