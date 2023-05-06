from flask import Flask, request, render_template
import math
#import pandas as pd
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
#from nltk import sent_tokenize, word_tokenize, PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
#from nltk.cluster.util import cosine_distance
#from torchmetrics.text.rouge import ROUGEScore
# from transformers import pipeline

app = Flask(__name__, template_folder='template')
# summarizer = pipeline("summarization")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['paragraph']
        summarized_text = summarize_extracive(input_text)
        return render_template('output.html', output_text=summarized_text)
    else:
        return render_template('input.html')

# def summarize_input(input_text):
#     # Summarize the input text using the Hugging Face Transformers library
#     # This example uses the GPT-2 model for summarization
#     summarized_text = summarizer(input_text, max_length=120, min_length=30, do_sample=False)
#     return summarized_text[0]['summary_text']

def summarize_extracive(input_text):
    #sentences = sent_tokenize(input_text)
    #stopwords = set(nltk.corpus.stopwords.words('english'))
    
    sentenceValueFrequencyBased = frequency_based(input_text)
    #sentenceValueTFIDF = dict()
    sentenceValueTFIDF = tf_idf(input_text)
    
    sentenceValueLSA = lsa_method(input_text)
    
    extractive_sumammry = extractive_sumammry_generation(sentenceValueFrequencyBased, sentenceValueTFIDF, sentenceValueLSA, input_text)
    
    return extractive_sumammry


def frequency_based(input_text):
    words = word_tokenize(input_text)
    freqTable = dict()
    sentences = sent_tokenize(input_text)
    #stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords = set('ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than')
    for word in words:
        word = word.lower()
        if word in stopwords:
            continue
        if word in freqTable:
            freqTable[word] +=1
        else:
            freqTable[word] = 1
    sentenceValueFrequencyBased = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValueFrequencyBased:
                    sentenceValueFrequencyBased[sentence] += freq
                else :
                    sentenceValueFrequencyBased[sentence] = freq
    return sentenceValueFrequencyBased


def tf_idf(input_text):
    sentences = sent_tokenize(input_text)
    total_documents = len(sentences)
    
    #2 Create the Frequency matrix of the words in each sentence
    frequency_matrix = {}
    #stopWords = stopwords
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopwords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table
        
        
    #3 Calculate TermFrequency and generate a matrix
    tf_matrix = {}

    for sent, f_table in frequency_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sent] = tf_table
        
    # 4 creating table for documents per words
    word_per_doc_table = {}
    for sent, f_table in frequency_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
        
    #return word_per_doc_table
    
    
    # 5 Calculate IDF and generate a matrix
    idf_matrix = {}

    for sent, f_table in frequency_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))

        idf_matrix[sent] = idf_table

    #return idf_matrix
    
    
    
    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    #return tf_idf_matrix
    
    
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    #return sentenceValue
    
    sentenceValueTFIDF = dict()
    i=0
    for sentence in sentenceValue:
      sentenceValueTFIDF[sentences[i]] = sentenceValue[sentence]
      i+=1
    
    
    return sentenceValueTFIDF
    
    
def lsa_method(input_text):
    
    #Split the given text to list of sentences and indentify title.
    titles = []
    sentences = sent_tokenize(input_text)
    document_list = sentences
    titles.append(input_text[0:min(len(input_text),5)])
    
    #For stemming purpose we use PorterStemmer()
    #Stemming is the process of reducing inflection word to its root form, i.e., stripping the suffix.
    
    tokenizer = RegexpTokenizer(r'\w+')
    #sW = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    texts = []
    for i in sentences:
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
      stopToken = [i for i in tokens if not i in stopwords]
      stemToken = [p_stemmer.stem(i) for i in stopToken]
      texts.append(stemToken)
    clean_doc = texts
    
    # Preparing corpus
    dict1 = corpora.Dictionary(clean_doc)
    doc_term_matrix = [dict1.doc2bow(doc) for doc in clean_doc] # Convert clean_doc to bow
    
    # creating LSA model
    num_of_topics = 2
    lsamodel = LsiModel(doc_term_matrix, num_topics = num_of_topics, id2word = dict1)
    #model = create_gensim_lsa_model(clean_doc,2)
    corpus_lsi = lsamodel[doc_term_matrix]
    
    
    vecSort = list(map(lambda i: list(), range(num_of_topics)))
    for i, docv in enumerate(corpus_lsi):
      for sc in docv:
        isent = (i,abs(sc[1]))
        vecSort[sc[0]].append(isent)
        
        
    sentenceValueLSA = dict()
    for i in vecSort[0]:
      sentenceValueLSA[sentences[i[0]]] = i[1]
      
    return sentenceValueLSA


def extractive_sumammry_generation(sentenceValueFrequencyBased, sentenceValueTFIDF, sentenceValueLSA, input_text):
    sentenceValueFrequencyBasedSorted = list(sorted(sentenceValueFrequencyBased.items(), key=lambda item: item[1], reverse = True))
    sentenceValueTFIDFSorted = list(sorted(sentenceValueTFIDF.items(), key=lambda item: item[1], reverse = True))
    sentenceValueLSASorted = list(sorted(sentenceValueLSA.items(), key=lambda item: item[1], reverse = True))
    
    #rouge = ROUGEScore()
    
    k = 4
    topSentencesFrequencyBased = selectTopSentencesFrequencyBased(k,sentenceValueFrequencyBasedSorted)
    topSentencesTFIDF = selectTopSentencesTFIDF(k,sentenceValueTFIDFSorted)
    topSentencesFrequencyLSA = selectTopSentencesLSA(k,sentenceValueLSASorted)
    
    
    topSentencesFrequencyBased = dict(topSentencesFrequencyBased)
    topSentencesTFIDF = dict(topSentencesTFIDF)
    topSentencesFrequencyLSA = dict(topSentencesFrequencyLSA)
    
    sentences = sent_tokenize(input_text)
    summaryFrequencyBased = ''
    for k in sentences:
      if k in topSentencesFrequencyBased:
        summaryFrequencyBased+=k
    
   # r1FB = np.array([rouge(summaryFrequencyBased, input_text)['rouge1_fmeasure']])
  #  r2FB = np.array([rouge(summaryFrequencyBased, input_text)['rouge2_fmeasure']])
   # rLFB = np.array([rouge(summaryFrequencyBased, input_text)['rougeL_fmeasure']])
    
    summaryTFIDF = ''
    for k in sentences:
      if k in topSentencesTFIDF:
        summaryTFIDF+=k
    
#    r1TF = np.array([rouge(summaryTFIDF, input_text)['rouge1_fmeasure']])
 #   r2TF = np.array([rouge(summaryTFIDF, input_text)['rouge2_fmeasure']])
   # rLTF = np.array([rouge(summaryTFIDF, input_text)['rougeL_fmeasure']])
    
    
    
    summaryLSA = ''
    for k in sentences:
      if k in topSentencesFrequencyLSA:
        summaryLSA+=k
    
    #r1LSA = np.array([rouge(summaryLSA, input_text)['rouge1_fmeasure']])
    #r2LSA = np.array([rouge(summaryLSA, input_text)['rouge2_fmeasure']])
   # rLLSA = np.array([rouge(summaryLSA, input_text)['rougeL_fmeasure']])
    rLFB=1
    rLTF=1
    rLLSA=1
    wFB = (rLFB/(rLFB+rLTF+rLLSA))*100
    wTF = (rLTF/(rLFB+rLTF+rLLSA))*100
    wLSA = (rLLSA/(rLFB+rLTF+rLLSA))*100
    
    hybridSentenceScoreWeights = dict()
    for i in sentenceValueFrequencyBased:
      hybridSentenceScoreWeights[i] = ((sentenceValueFrequencyBased[i]*wFB)+(sentenceValueTFIDF[i]*wTF)+(sentenceValueLSA[i]*wLSA))/100
    
    hybridSentenceScoreWeightsSorted = list(sorted(hybridSentenceScoreWeights.items(), key=lambda item: item[1], reverse = True))
    
    topSentencesWithWeights = selectTopSentencesWeights(k, hybridSentenceScoreWeightsSorted)
    topSentencesWithWeights = dict(topSentencesWithWeights)


    summaryWeights = ''
    for k in sentences:
      if k in topSentencesWithWeights:
        summaryWeights+=k+"\n"
    
    return summaryWeights


def selectTopSentencesFrequencyBased(k, sentenceValueFrequencyBasedSorted):
  return sentenceValueFrequencyBasedSorted[:k]

def selectTopSentencesTFIDF(k, sentenceValueTFIDFSorted):
  return sentenceValueTFIDFSorted[:k]

def selectTopSentencesLSA(k, sentenceValueLSASorted):
  return sentenceValueLSASorted[:k]
    
def selectTopSentencesWeights(k, hybridSentenceScoreWeightsSorted):
  return hybridSentenceScoreWeightsSorted[:k]
    

if __name__ == '__main__':
    app.run()

















