import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.dates
from matplotlib import pyplot as plt
import pickle

def linear(df, brand, model=None): 
	X_train, X_test, y_train, y_test = split_df(df, brand)

	if model is None:
		model = LinearRegression()
		model.fit(X_train, y_train)

	r_sq = model.score(X_train, y_train)
	y_pred = model.predict(X_train)
	intercept = model.intercept_
	slope = model.coef_[0]
	# best_fit = X_train[:,0] * slope + intercept
	fig = plt.figure(figsize=(9,6))
	ax = plt.gca()
	ax.plot(X_train[:,0],y_train,'o')
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
	plt.plot(X_train, y_pred, 'k-', color = "r")
	# plt.xticks(rotation='vertical')
	plt.title('Линейная регрессия бренда: ' + brand + '\nr_sq: %f' % r_sq)
	plt.xlabel('Дата')
	plt.ylabel('Цена')
	# plt.show()
	plt.rcParams['pdf.fonttype'] = 42
	plt.rcParams['font.family'] = 'Calibri'

	fileimg = 'linear.png'
	plt.savefig('./static/linear.png')
	plt.savefig('./static/linear.pdf')

	filemodel = brand+'.sav'
	pickle.dump(model, open('./static/models/'+filemodel, 'wb'))

	y_pred_test = model.predict(X_test)

	return y_test, y_pred_test, fileimg, filemodel

def polynom(df, brand):
	X_train, X_test, y_train, y_test = split_df(df, brand)
	poly_reg = PolynomialFeatures(degree=8)
	X_poly = poly_reg.fit_transform(X_train)
	pol_reg = LinearRegression()
	pol_reg.fit(X_poly, y_train)
	y_pred = pol_reg.predict(poly_reg.fit_transform(X_train))
	r_sq = pol_reg.score(X_poly, y_train)

	fig = plt.figure(figsize=(9,6))
	ax = plt.gca()
	ax.plot(X_train, y_train,'o')
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
	plt.plot(X_train, y_pred, 'k-', color = "r")
	# plt.xticks(rotation='vertical')
	plt.title('Полиномиальная регрессия бренда: ' + brand + ' \nr_sq:  %f' % r_sq)
	plt.xlabel('Дата')
	plt.ylabel('Цена')

	filename = 'polynom.png'
	plt.savefig('./static/'+filename)
	plt.savefig('./static/polynom.pdf')

	return filename


def split_df(df, brand):
	df = df[df['brand']==brand]
	df['dateUpdated'] = pd.to_datetime(df.dateUpdated) 
	# df = df.sort_values(by='dateUpdated')
	df['dateUpdated']=pd.to_datetime(df.dateUpdated)
	df['dateUpdated']=df['dateUpdated'].dt.normalize()
	data = df.groupby('dateUpdated')['prices.amountMax'].mean()
	# date_brands = df['dateUpdated']
	# price_brands = df['prices.amountMax']

	# Даты, которые будут отложены по оси X
	# xdata = date_brands
	date_brands = data.index
	# Данные, которые будут отложены по оси Y
	# ydata = price_brands
	price_brands = data.values

	X = np.array(matplotlib.dates.date2num(date_brands)).reshape((-1, 1))
	y = np.array(price_brands)
	row_count = X.shape[0]
	split_point = X.shape[0]-int(row_count*1/5)
	X_train = X[:split_point]
	X_test = X[split_point:]
	y_train = y[:split_point]
	y_test = y[split_point:]

	return X_train, X_test, y_train, y_test