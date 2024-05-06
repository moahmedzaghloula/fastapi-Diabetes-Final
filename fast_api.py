from fastapi import FastAPI, Form
import joblib

app = FastAPI()

model = joblib.load('diabetes.pkl')


@app.get("/")  
async def home():
    return {"Project Name": "Diabetes Final"}


@app.post("/predict")
async def predict(pregnancies: int = Form(...),
                  glucose: int = Form(...),
                  blood_pressure: int = Form(...),
                  skin_thickness: int = Form(...),
                  insulin: int = Form(...),
                  bmi: float = Form(...),
                  diabetes_pedigree_function: float = Form(...),
                  age: int = Form(...)):
    prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    if prediction[0] == 0:
        result = 'Non-Diabetic'
    else:
        result = 'Diabetic'

    return {"result": result}
