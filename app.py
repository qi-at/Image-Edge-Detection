from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import os
from io import BytesIO
import time

app = Flask(__name__)

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    img_file = request.files['img_file']
    
    img_bytes = img_file.read()
    
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    technique = request.form['technique']
    
    if technique == 'canny':
        edges = cv2.Canny(img, 100, 200)
    elif technique == 'prewitt':
        kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        edges = cv2.filter2D(img, -1, kernel)
    elif technique == 'robert':
        kernel = np.array([[1,0],[0,-1]])
        edges = cv2.filter2D(img, -1, kernel)
    elif technique == 'sobel':
        edges_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
    elif technique == 'global':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    elif technique == 'otsu':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif technique == 'kmeans':
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        edges = res.reshape((img.shape))
    
    timestamp = int(time.time())
    file_name = f'edges_{timestamp}.png'
    file_path = os.path.join(uploads_dir, file_name)
    
    cv2.imwrite(file_path, edges)
    
    _, buffer = cv2.imencode('.png', edges)
    edges_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return render_template('result.html', edges_base64=edges_base64)

if __name__ == '__main__':
    app.run(debug=True)