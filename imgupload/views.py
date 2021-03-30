from django.shortcuts import render
from .forms import ImageUploadForm        
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
def handle_uploaded_file(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

# Create your views here.
def home(request):

    return render(request,'home.html')
def imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        model = ResNet50(weights='imagenet')
        img = 'img.jpg'

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print(preds)
        print('Predicted:', decode_predictions(preds, top=3)[0])
    return render(request,'result.html')
