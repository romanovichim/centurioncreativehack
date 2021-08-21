
# https://pypi.org/project/SpeechRecognition/
import os 
import moviepy.editor as mp
#https://zulko.github.io/moviepy/
from nnsplit import NNSplit
#https://bminixhofer.github.io/nnsplit/
from vosk import Model, KaldiRecognizer
import sys
import wave
import json


import speech_recognition as sr
from itertools import combinations
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
import networkx as nx

#model = Model("model-ru-big")


def audiototext(AUDIO_FILE):
    wf = wave.open(AUDIO_FILE, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    s = ""
    
    while True:
        data = wf.readframes(1000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            s = s + " " + json.loads(rec.Result())['text']
        else:
            #print(rec.PartialResult())
            pass

    
    s = s + " "+ json.loads(rec.FinalResult())['text']

    return s

def audiototextg(AUDIO_FILE):
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    return r.recognize_google(audio, language="ru_RU")
    
def wavtotext(audioname):
    # сделать доп обработку со своим ключом если что-то не сработает в таком
    #делает из wav текст возвращает флаг success/error и текст
    
    AUDIO_FILE = os.path.join(os.path.abspath(os.getcwd()),"files","{0}".format(audioname),"{0}.wav".format(audioname))

    try:
        return "success",audiototextg(AUDIO_FILE)
    except sr.UnknownValueError:
        return "error","Could not understand audio"
    except sr.RequestError as e:
        return "error","Could not request results if API is used service;"
     
#result = wavtotext("russian.wav")
#print(result[0]) # флаг
#print(result[1]) # текст

def mp4totext(videoname):
    #перевод видео а так же всякие удаления файлов
    try:
       # Insert Local Video File Path
        VIDEO_FILE = os.path.join(os.path.abspath(os.getcwd()),"files","{0}".format(videoname),"{0}.mp4".format(videoname))
        clip = mp.VideoFileClip(VIDEO_FILE)
        # Insert Local Audio File Path
        AUDIO_FILE = os.path.join(os.path.abspath(os.getcwd()),"files","{0}".format(videoname),"{0}.wav".format(videoname))
        clip.audio.write_audiofile(AUDIO_FILE,codec='pcm_s16le', verbose=False,logger=None)
        #videoname is ggenerated id for files of one session
        result =  wavtotext(videoname)
        # закрыть процесс
        clip.close()
        #del video
  
        if os.path.exists(VIDEO_FILE):
          os.remove(VIDEO_FILE)
        #del audio
        if os.path.exists(AUDIO_FILE):
          os.remove(AUDIO_FILE)

        folder = os.path.join(os.path.abspath(os.getcwd()),"files","{0}".format(videoname))
        if os.path.exists(folder):
          os.rmdir(folder)
        #del folder

        return result[0],result[1]
    except Exception as e: # work on python 3.x
        return "error","Video processing problem; {0}".format(str(e))

#print(mp4totext("test"))






def add_punkt(text):
    ru_model = os.path.join(os.path.abspath(os.getcwd()),"models","ru","model.onnx")
    splitter = NNSplit(ru_model)
    # returns `Split` objects
    splits = splitter.split([text])[0]
    # a `Split` can be iterated over to yield smaller splits or stringified with `str(...)`.
    text_punkt = ""
    for sentence in splits:
        text_punkt = text_punkt + str(sentence).rstrip() + ". "

    return text_punkt



from itertools import combinations
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
import networkx as nx

def similarity(s1, s2):
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))

# Выдает список предложений отсортированных по значимости
def textrank(text):
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    lmtzr = RussianStemmer()
    words = [set(lmtzr.stem(word) for word in tokenizer.tokenize(sentence.lower()))
             for sentence in sentences] 	 
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(words[i], words[j])) for i, j in pairs]
    scores = filter(lambda x: x[2], scores)
    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    pr = nx.pagerank(g)
    return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr), key=lambda x: pr[x[0]], reverse=True)

# Сокращает текст до нескольких наиболее важных предложений
def sumextract(text, n=5):
    tr = textrank(text)
    top_n = sorted(tr[:n])
    return ' '.join(x[2] for x in top_n)


text_with_punkt =  "Один из универсальных рецептов стартапа - зумеры изобрели икс. Вот например давным-давно существуют локальные пиццерии работающие только на доставку. Клиент делает заказ по телефону и ее приносит курьер. Эту незамысловатую бизнес-модель используют и мелкие игроки с одной точкой и франшизы и федеральные сети. Никаких инноваций на рынке не происходит десятилетиями. да они и не нужны клиенты и так довольны. А потом мы добавляем сюда няшное мобильное приложение и венчурное финансирование - и вот вместо доставки пиццы появляются уже темные кухни. Прибыль в моменте им не нужна, за всё платят инвесторы, так что ингредиенты в рецептах используются чуть качественнее, а курьеры лучше говорят по-русски, но в целом суть бизнеса не изменилась ни на йоту. Зато стартап. Часто - успешный. Гренки не могут оцениваться по 10 выручек, а крутоны могут.Инвесторам нужно рассказывать о будущем искусственном интеллекте, который заменит операторов, и эффекте масштаба - только мы накопим нужное количество данных. Венчурные деньги позволят давать скидки и заказчикам с традиционными тендерами. Кстати, потребуется и реальная разработка - какой-то универсальный внутренний интерфейс, клики по которому будут транслироваться во взаимодействие с админ-панелями клиента. Капитализация миллиардов двадцать долларов вполне реальна, если быть первым и двигаться быстро."

text_with_punkt = text_with_punkt + text_with_punkt


from nltk import tokenize
import nltk
import string
exclude = set(string.punctuation)
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
#векторное пространство
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Латентное распределние Дирихле 
from sklearn.decomposition import LatentDirichletAllocation


stop = ['и', 'в', 'во', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']

def normalize(word):
  return morph.parse(word)[0].normal_form

def clean(doc):
    #lower + split
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # remove any stp words present
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # remvoe punkt and normalize 
    normalized = " ".join(normalize(word) for word in punc_free.split())
    return normalized


def cluster(text_with_punkt):
    sent_text = tokenize.sent_tokenize(text_with_punkt)
    clean_corpus = [clean(doc).split() for doc in sent_text]

    tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

    tf_idf_arr = tf_idf_vectorizer.fit_transform(clean_corpus)
    cv_arr = cv_vectorizer.fit_transform(clean_corpus)

    vocab_tf_idf = tf_idf_vectorizer.get_feature_names()

    #пока делим просто на 3
    lda_model = LatentDirichletAllocation(n_components=3, max_iter = 20, random_state=20)

    X_topics = lda_model.fit_transform(tf_idf_arr)

    topic_words = lda_model.n_components
    doc_topic = lda_model.transform(tf_idf_arr)

    # соберем все в массив
    result = []
    for n,i in  zip(range(doc_topic.shape[0]),sent_text):
        topic_most_pr = doc_topic[n].argmax()
        #print("doc: {} topic: {}\n".format(n,topic_most_pr),i)
        temp=[]
        temp.append(i)
        temp.append(topic_most_pr)
        result.append(temp)

    return result





def clustersum(text_with_punkt):
    result = cluster(text_with_punkt)
    # выведем темам
    itog=""
    for i in range(3):
      tt = ""
      for sent in result:
        if(sent[1] == i ):
          tt += sent[0]

      
      itog += sumextract(tt,2)
      

    return itog

def howto(text_with_punkt):
    #выбирает надо ли кластеризировать
    sent_text = tokenize.sent_tokenize(text_with_punkt)
    if(len(sent_text)<8):
        return sumextract(text, 2)
    if(len(sent_text)<16):
        return sumextract(text, 3)
    if(len(sent_text)>15):
        return clustersum(text_with_punkt)
        


