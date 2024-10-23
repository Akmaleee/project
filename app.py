import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model (Assuming it's VGG19)
def load_model():
    vgg19_model = models.vgg19(pretrained=True)
    vgg19_model.classifier[6] = nn.Linear(4096, 1)  # Binary classification
    vgg19_model.load_state_dict(torch.load('model/vgg19Model.h5', map_location=device))
    vgg19_model.to(device)
    vgg19_model.eval()
    return vgg19_model

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load and preprocess image
            img = Image.open(file_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0).to(device)

            # Model prediction
            output = model(batch_t)
            pred = torch.sigmoid(output).item()
            result = 'Benign' if pred < 0.5 else 'Malignant'

            return render_template('result.html', result=result, probability=f"{pred:.4f}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
