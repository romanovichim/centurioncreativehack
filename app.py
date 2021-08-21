import os
from flask import Flask, request, redirect, url_for, send_from_directory,after_this_request,render_template
from werkzeug.utils import secure_filename
#from util import removefile
import uuid


from sr import mp4totext
from sr import add_punkt
from sr import sumextract



ALLOWED_EXTENSIONS = set(['mp4'])
# куда и какие расширения для ограничений


app = Flask(__name__, instance_path=os.path.dirname(os.path.realpath(__file__)))
UPLOAD_FOLDER = os.path.join(app.instance_path, 'files')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# включаем и настраиваем папку

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# проверка расширения файла

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist("file[]")
        #сгенерируем уникальный идентификатор
        unique= uuid.uuid4().hex
        os.mkdir(app.config['UPLOAD_FOLDER']+'/'+unique+'/')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER']+'/'+unique, unique+".mp4"))

        #return render_template('success.html')

        return redirect(url_for('recog_file',filenames=unique))



    return render_template('index.html')
   



@app.route('/recog/<filenames>', methods=['GET', 'POST'])
def recog_file(filenames):
    if request.method == 'POST':
        #return 'You entered: {}'.format(request.form['text'])
        text = request.form['text']
        return redirect(url_for('punkt_file',filenames=filenames,text = text))
    
    #table= generatetopics(filenames)
    #return render_template('trytable.html', tbl=table,bgimgname=filenames+".png")
    text = mp4totext(filenames)[1]
    #sumextract(add_punkt(text), 2)
    return render_template('recog.html', text_rdy = text)
    #return render_template('trytable.html', tbl=table)
    #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    
@app.route('/punkt/<filenames>/<text>', methods=['GET', 'POST'])
def punkt_file(filenames,text):
    if request.method == 'POST':
        #return 'You entered: {}'.format(request.form['text'])
        text = request.form['text']
        return redirect(url_for('summary',filenames=filenames,text = text))
    
    text = add_punkt(text)
    return render_template('punkt.html', text_rdy = text)

@app.route('/summary/<filenames>/<text>')
def summary(filenames,text):
    text = sumextract(text, 2)
    return render_template('sum.html', text_rdy = text)



if __name__ == '__main__':
    app.run()
