import time
st = time.time()

from django.conf import settings as django_settings

# import os
from os import system, path as os_path

from numpy import array, log10
from keras.models import Model,load_model # load_model for loading saved models after fitting
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img # To load image and convert or resize
from keras.preprocessing.image import img_to_array # To change image to numpy arrays
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from nltk.translate.bleu_score import corpus_bleu # For scoring our model

from pickle import load as pickle_load
from tensorflow import Graph, Session

print('Module load time: %s' % str(time.time() - st))

graph2 = Graph()
with graph2.as_default():
  session2 = Session()
  with session2.as_default():
    VGG_MODEL = VGG16()
    VGG_MODEL.layers.pop()

def extract_features(filename):
  with graph2.as_default():
    with session2.as_default():
      FEATURE_EXT_MODEL = Model(inputs=VGG_MODEL.inputs, outputs=VGG_MODEL.layers[-1].output)
      # load the photo
      image = load_img(filename, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = preprocess_input(image)
      # get features
      feature = FEATURE_EXT_MODEL.predict(image, verbose=0)
      return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def beam_search(model, tokenizer, photo, max_length, beam_size = 3):
  candidates = []
  log_probabilities_of_candidates = [] # Normalized on length of candidate
  in_text = 'startseq'
  candidates.append(in_text)
  log_probabilities_of_candidates.append(0) # as log(1) is 0
  for i in range(max_length):
    #print(candidates)
    new_candidates = []
    log_probabilities_of_new_candidates = [] # Normalized on length of candidate
    for index in range(len(candidates)):

      # Extracting info of current candidate
      candidate = candidates[index]
      current_prob = log_probabilities_of_candidates[index]

      # If current candidate has observed endseq, then directly push it to new candidates with existing probability and move to next candidate
      if candidate.split()[-1] == 'endseq':
        new_candidates.append(candidate)
        log_probabilities_of_new_candidates.append(current_prob)
        continue

      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([candidate])[0]

      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)

      # predict next word
      yhat = model.predict([photo,sequence], verbose=0)

      # Finding probable children candidates
      ids_of_children_candidates = yhat.argsort()[0][::-1][0:beam_size]

      #Slicing probabilities of these children candidates from prediction array
      probabilities = log10(yhat[0][ids_of_children_candidates])

      # Pushing new children candidate sentences
      length_of_candidate  = len(candidate.split())
      for j in range(len(ids_of_children_candidates)):
        word_id = ids_of_children_candidates[j]
        new_children_candidate = candidate + ' ' + word_for_id(word_id, tokenizer)
        # Normalizing probabilities
        new_children_candidate_probability =  probabilities[j] + (current_prob * length_of_candidate) # first un-normalize prev probability
        new_children_candidate_probability = (new_children_candidate_probability)/(length_of_candidate + 1) # normalize with updated length

        # append results to new candidates list
        new_candidates.append(new_children_candidate)
        log_probabilities_of_new_candidates.append(new_children_candidate_probability)

    # Now sort all candidates on basis of their probabilities and consider only top beam_size candidates
    temp = []
    #print(new_candidates)
    for j in range(len(new_candidates)):
      temp.append((log_probabilities_of_new_candidates[j], new_candidates[j]))

    top_candidates = sorted(temp, reverse = True)[0:beam_size]

    # Update lists of candidates and their probabilities
    candidates = []
    log_probabilities_of_candidates = []
    for j in range(beam_size):
      candidates.append(top_candidates[j][1])
      log_probabilities_of_candidates.append(top_candidates[j][0])
  return candidates[0]

def predict_from_link(link):
  system("curl " + link + " > ImageCaptioning/ImageCaptioning/Data/test_image.jpg")
  st = time.time()
  model = load_model("ImageCaptioning/ImageCaptioning/Data/final_model.h5")
  print('Model load time %s' % str(time.time() - st))
  st = time.time()
  tokenizer = pickle_load(open('ImageCaptioning/ImageCaptioning/Data/tokenizer.pkl', 'rb'))
  print('Tokenizer load time %s' % str(time.time() - st))
  max_length = 33
  st = time.time()
  photo = extract_features('ImageCaptioning/ImageCaptioning/Data/test_image.jpg')
  print('Extraction time %s' % str(time.time() - st))
  st = time.time()
  description = beam_search(model, tokenizer, photo, max_length, beam_size=5)
  print('Beam Search Timt %s' % str(time.time() - st))
  description = description.split(" ")[1:-1]
  return " ".join(description)

MEDIA_ROOT = django_settings.MEDIA_ROOT
DATA_DIR = os_path.join(django_settings.BASE_DIR, 'ImageCaptioning/Data')
MODEL_PATH = os_path.join(DATA_DIR, 'final_model.h5')
TOKENIZER_PATH = os_path.join(DATA_DIR, 'tokenizer.pkl')
graph1 = Graph()
with graph1.as_default():
    session1 = Session()
    with session1.as_default():
      MODEL = load_model(MODEL_PATH)
      TOKENIZER = pickle_load(open(TOKENIZER_PATH, 'rb'))

def predict_from_storage(media_image_path):
  with graph1.as_default():
    with session1.as_default():
      EXACT_IMAGE_PATH = os_path.join(MEDIA_ROOT, media_image_path)
      MAX_LENGTH = 33
      photo = extract_features(EXACT_IMAGE_PATH)
      description = beam_search(MODEL, TOKENIZER, photo, MAX_LENGTH, beam_size=5)
      description = " ".join(description.split(" ")[1:-1])
      return description
