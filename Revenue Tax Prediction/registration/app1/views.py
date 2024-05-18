from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from joblib import load
import pandas as pd
from django.http import JsonResponse
# Create your views here.


def HomePage(request):
    return render (request,'home.html')

def predict_revenue(request):
    if request.method == 'POST':
        # Receive input from the user
        revenue_category = request.POST.get('revenue_category')
        year_str = request.POST.get('year')

        # Check if the year field is empty
        if not year_str:
            return JsonResponse({'error': 'Year field is empty'})

        try:
            year = int(year_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid year value'})

        # Load the precompiled model, scaler, and label encoder
        model = load('trained_model.joblib')
        scaler = load('scaler.joblib')
        label_encoder = load('label_encoder.joblib')

        # Preprocess the input data
        revenue_category_encoded = label_encoder.transform([revenue_category])

        # Create a DataFrame with the preprocessed input data
        new_data = pd.DataFrame({'Revenue category': revenue_category_encoded[0], 'Year': year}, index=[0])

        # Scale the numerical features (e.g., 'Year') using the StandardScaler
        new_data_scaled = scaler.transform(new_data)

        # Make predictions using the precompiled model
        predicted_value = model.predict(new_data_scaled)

        # Return the predicted value as JSON response
        return JsonResponse({'predicted_value': predicted_value.tolist()})

    
