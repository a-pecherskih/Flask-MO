import pandas as pd
import os
from flask import Flask, make_response, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
import pdfkit
import prediction
import pickle
import PyPDF2

#env\Scripts\activate
#flask run
#diactivate
app = Flask(__name__, static_url_path='/static')


UPLOAD_FOLDER = './uploads'
FILENAME = './uploads/data.csv'
ALLOWED_EXTENSIONS = set(['csv', 'sav'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/index')
@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')
		

@app.route('/pdf')
def create_pdf():
	df = pd.read_csv(FILENAME)
	mean_brands = df.groupby('brand')['prices.amountMax'].mean()
	path_wkthmltopdf = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'

	config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
	rendered = render_template('pdf.html', mean_brands=mean_brands)
	pdf = pdfkit.from_string(rendered, False, configuration=config)

	response = make_response(pdf)
	response.headers['Content-Type'] = 'application/pdf'
	response.headers['Content-Disposition'] = 'inline'
	return response


@app.route('/meancount', methods=['POST','GET'])
def mean_count():
	df = pd.read_csv(FILENAME)
	brands = df.brand.unique()
	max_price = df['prices.amountMax'].max()
	max_brand = df[df['prices.amountMax']==df['prices.amountMax'].max()]['brand'].unique()
	if request.method == 'POST':
		brand = request.form['brand']
		mean = round(df[df['brand']==brand]['prices.amountMax'].mean(),2)
		return render_template('mean_count.html', brands=brands, mean_price=mean, ibrand=brand, max_brand=max_brand[0], max_price=max_price)
	else:
		brand = df['brand'][0]
		mean = round(df[df['brand']==brand]['prices.amountMax'].mean(),2)
	return render_template('mean_count.html', brands=brands, mean_price=mean, ibrand=brand, max_brand=max_brand[0], max_price=max_price)


#FILE
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadfile', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename("data.csv")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('mean_count'))
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	df = pd.read_csv(FILENAME)
	brands = df['brand'].value_counts().loc[lambda x : x>500].index
	if request.method == 'POST':
		brand = request.form['brand']
		y_test, y_pred, img_linear, filemodel = prediction.linear(df, brand)
		img_polynom = prediction.polynom(df, brand)
		predict_pdf(brand, y_test, y_pred)
		generate_pdf()
		return render_template('predict.html', brands=brands, ibrand=brand, img_linear=img_linear, img_polynom=img_polynom, filemodel=filemodel)
	else:
		brand = brands[0]
		return render_template('predict.html', brands=brands, ibrand=brand)

@app.route('/load_regression', methods=['POST'])
def load_regression():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	df = pd.read_csv(FILENAME)
	brands = df['brand'].value_counts().loc[lambda x : x>500].index
	# brand = request.form['brand']
	brand = filename.rsplit('.',1)[0].replace("_", " ")
	model = pickle.load(open('./uploads/'+filename, 'rb'))
	y_test, y_pred, img_linear, filemodel = prediction.linear(df, brand, model)
	img_polynom = prediction.polynom(df, brand)
	predict_pdf(brand, y_test, y_pred)
	generate_pdf()
	return render_template('predict.html', brands=brands, ibrand=brand, img_linear=img_linear, img_polynom=img_polynom, filemodel=filemodel)

def predict_pdf(brand, y_test, y_pred):
	path_wkthmltopdf = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'

	config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
	rendered = render_template('predict_pdf.html', y_test=y_test, y_pred=y_pred, len=len(y_pred))
	pdf = pdfkit.from_string(rendered, './static/predict.pdf', configuration=config)


def generate_pdf():
    #Получаем список файлов в переменную files 
    file_img_linear = 'linear.pdf'
    file_img_polynom = 'polynom.pdf'
    file_predict = 'predict.pdf'
    files = [file_img_linear, file_img_polynom, file_predict]
    merger = PyPDF2.PdfFileMerger()
    
    for filename in files:
        merger.append(fileobj=open(os.path.join('./static',filename),'rb'))
    
    merger.write(open(os.path.join('./static','out.pdf'), 'wb'))


