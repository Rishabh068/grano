from flask import Flask,render_template,request
import crop_recommend as cp
import requests
import numpy as np
import pickle
crop_recommendation_model_path = 'RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


app= Flask(__name__)


@ app.route('/',methods=["GET","POST"])
def home():
    title = 'GRANO - Crop Recommendation'
    return render_template('form.html', title=title)

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'GRANO - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form.get("nitrogen"))
        P = int(request.form.get("phosphorous"))
        K = int(request.form.get("potassium"))
        ph =float(request.form.get("PH"))
        rainfall =float(request.form.get("Rainfall"))
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
    data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]

    return render_template('result.html',prediction=final_prediction ,title=title)


if __name__=='__main__'  :
    app.run()