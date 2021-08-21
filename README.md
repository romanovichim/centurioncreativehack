#Функциональность 
загрузка файла mp4
перевод mp4 в wav
распознавание текста в wav
проставление пунктуации в массиве слов
topic modelling с помощью Латентного распределения Дирихле
алгоритм textrank
#Стек
python, flask, sklearn, nnsplit, Kaldi, networkx
#Среда
локально на Windows 10
#Установка
run: pip install -r requirements.txt in your shell
#Особенности
для распознования с помощью Kaldi необходимо скачать модель:
vosk-model-ru-0.10.zip
из https://alphacephei.com/vosk/models 
и разархивировать в папку model-ru-big
а также в файле sr раскомментировать строчку 21, а также удалить букву g в строке 58
базово работает через гугл апи для простоты развертывания
#Пример файла для распознования
https://drive.google.com/drive/folders/1gVjcxkULXLdYNz_Ucm_fKw-fY3SvSQc5?usp=sharing
