from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# Load the model
model = CustomResNet().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the transformation
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image=np.array(image))['image'].unsqueeze(0)
    return image.cuda()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    image = transform_image(file.read())
    with torch.no_grad():
        outputs = model(image)
    prediction = torch.sigmoid(outputs).item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
