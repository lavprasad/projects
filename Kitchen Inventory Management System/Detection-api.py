from clarifai.rest import ClarifaiApp

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'b03bedc05cf14099bc6668bc5533add1'

try:
    app = ClarifaiApp(api_key=api_key)
    print("Clarifai API connection successful!")

    # Define the model you want to use
    model = app.public_models.general_model

    # List of objects you want to detect
    objects_to_detect = ['chicken', 'karela', 'tomato', 'garlic', 'ginger', 'carrot', 'potato', 'lemon']

    # URL or local path to the image you want to analyze
    image_url = 'C:\\Users\\Asus\\Downloads\\apple.jpg'

    # Perform prediction
    response = model.predict_by_filename(image_url)

    # Extract predictions
    predictions = response['outputs'][0]['data']['concepts']

    # Print the raw predictions
    print("Raw Predictions:")
    print(predictions)

    # Filter predictions for objects of interest
    detected_objects = [obj['name'] for obj in predictions if obj['name'] in objects_to_detect]

    # Print the filtered predictions
    print("Detected Objects:")
    print(detected_objects)

except Exception as e:
    print(f"Error initializing ClarifaiApp: {e}")
