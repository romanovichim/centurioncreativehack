
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

    
def wavtotext(audioname):
    # сделать доп обработку со своим ключом если что-то не сработает в таком
    #делает из wav текст возвращает флаг success/error и текст
    
    AUDIO_FILE = os.path.join(os.path.abspath(os.getcwd()),"files","{0}".format(audioname),"{0}.wav".format(audioname))

    try:
        return "success",audiototext(AUDIO_FILE)
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
