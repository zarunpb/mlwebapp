from django.shortcuts import render

# Create your views here.
import pandas as pd
from django.http import JsonResponse
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        df = pd.read_excel(uploaded_file)
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        dump(model, 'model.pkl')
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return JsonResponse({'accuracy': accuracy})
    return render(request, 'train.html')

def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        df = pd.read_excel(uploaded_file)
        
        model = load('model.pkl')
        
        predictions = model.predict(df)
        
        return JsonResponse({'predictions': predictions.tolist()})
    return render(request, 'predict.html')
