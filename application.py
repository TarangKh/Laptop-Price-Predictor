from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import Ridge
import pickle

application = Flask(__name__)
app = application
laptops = pd.read_csv("laptop_data_cleaned.csv")



@app.route("/")
def index():
    brand = laptops["brand"].unique()
    model = laptops["model"].unique()
    processor_brand = laptops["processor_brand"].unique()
    processor_name = laptops["processor_name"].unique()
    ram_type = laptops["ram_type"].unique()
    os = laptops["os"].unique()


    return render_template("index.html", brand=brand, model=model, processor_brand=processor_brand, processor_name = processor_name, ram_type = ram_type, os=os)



@app.route("/predict", methods=["POST"])
def predict():
    try:
        brand = request.form.get("brand")
        model_name = request.form.get("model")
        processor_brand = request.form.get("processor_brand")
        processor_name = request.form.get("processor_name")
        gen = int(request.form.get("gen"))
        ram = int(request.form.get("ram"))
        ram_type  = request.form.get("ram_type")
        ssd = int(request.form.get("ssd"))
        hdd = int(request.form.get("hdd"))
        os = request.form.get("os")
        os_bit = int(request.form.get("os_bit"))
        gpu = int(request.form.get("gpu"))
        Touchscreen = int(request.form.get("Touchscreen"))
        model = pickle.load(open("model.pkl", "rb"))
        prediction = model.predict(pd.DataFrame([[brand, model_name, processor_brand, processor_name, gen, ram, ram_type, ssd, hdd, os, os_bit, gpu, Touchscreen]], columns=['brand', 'model', 'processor_brand', 'processor_name',
        'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit',
        'graphic_card_gb', 'Touchscreen']))
        # prediction = model.predict(pd.DataFrame([["Lenovo", "A6-9225", "AMD", "A6-9225 Processor", 10, 4, "DDR4", 256, 1024, "Windows", 64, 2, 1]],
        #                           columns=['brand', 'model', 'processor_brand', 'processor_name',
        #    'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit',
        #    'graphic_card_gb', 'Touchscreen']))
        # print([brand, model_name, processor_brand, processor_name, gen, ram, ram_type, ssd, hdd, os, os_bit, gpu, Touchscreen])
        
        return str(int(prediction[0]))
    
    except (ValueError):
        return "ValueError"
    except Exception as e:
        return render_template("exception.html", e=e)

        

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port="5000")