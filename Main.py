# develop a classifier for the 5 Celebrity Faces Dataset
import os
from os.path import join, dirname, realpath
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from PIL import Image
from matplotlib import pyplot
from numpy import asarray
from sklearn.metrics import accuracy_score
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from werkzeug.utils import secure_filename
from flask import Flask,request,abort,json
from flask_cors import CORS

app=Flask(__name__)

CORS(app)
def extract_image(image):
  img1 = Image.open(image)
  img1 = img1.convert('RGB')
  pixels = asarray(img1)
  detector = MTCNN()
  f = detector.detect_faces(pixels)
  x1,y1,w,h = f[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1+w)
  y2 = abs(y1+h)
  store_face = pixels[y1:y2,x1:x2]
  image1 = Image.fromarray(store_face,'RGB')
  image1 = image1.resize((160,160))
  face_array = asarray(image1)
  return face_array
def extract_embeddings(model,face_pixels):
  face_pixels = face_pixels.astype('float32')
  mean = face_pixels.mean()
  std  = face_pixels.std()
  face_pixels = (face_pixels - mean)/std
  samples = expand_dims(face_pixels,axis=0)
  yhat = model.predict(samples)
  return yhat[0]
####
def predict_face(Img):
  face = extract_image(Img)
  testx = asarray(face)
  testx = testx.reshape(-1,160,160,3)
  # print("Input test data shape: ",testx.shape)

  #find embeddings
  model = load_model('input/facenet-keras/facenet_keras.h5',compile=False)
  new_testx = list()
  for test_pixels in testx:
    embeddings = extract_embeddings(model,test_pixels)
    new_testx.append(embeddings)
  new_testx = asarray(new_testx)  
  # print("Input test embedding shape: ",new_testx.shape)
  # print(new_testx)
  # load faces
  data = load('Students.npz')
  testX_faces = data['arr_2']
  # load face embeddings
  data = load('Students-faces-embeddings.npz')
  trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
  # normalize input vectors
  in_encoder = Normalizer(norm='l2')
  trainX = in_encoder.transform(trainX)
  new_testx = in_encoder.transform(new_testx)
  # label encode targets
  out_encoder = LabelEncoder()
  new_testy=trainy
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  new_testy = out_encoder.transform(new_testy)
  # fit model
  model = SVC(kernel='linear', probability=True)
  model.fit(trainX, trainy)
  # yhat_class = model.predict(trainX)
  predict_test=model.predict(new_testx)
  # confidence = max(predict_test)

  # get name
  #Accuracy
  # acc_train = accuracy_score(trainy,yhat_class)

  #display
  # trainy_list = list(trainy)
  # p=int(predict_test)
  # if p in trainy_list:
  #   val = trainy_list.index(p)
    
    
  predict_test = out_encoder.inverse_transform(predict_test)
  print(predict_test[0])
  return predict_test[0]
UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads/')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict',methods=['GET','POST'])
def predict():
  if request.method=='GET' or request.method=='POST':
      try:
        if 'file' not in request.files:
          return 'No File Uploaded',400
        else:
          Img=request.files['file']
          if Img.filename == '':
            return 'No File Name',400
          if Img and allowed_file(Img.filename):
            filename = secure_filename(Img.filename)
            Img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            result=predict_face('uploads/'+filename)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            response = app.response_class(
            response=json.dumps(result),
            status=200,
            mimetype='application/json') 
        return response
      except Exception as e:
        print(e)
        return 'failed',400
#plt.imshow(Img)

if __name__ == '__main__':
    app.run()      


# class_index = yhat_class[0]
# class_probability = yhat_prob[0,class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# # plot for fun
# pyplot.imshow(face)
# title = '%s (%.3f)' % (predict_names[0], class_probability)
# pyplot.title(title)
# pyplot.show()