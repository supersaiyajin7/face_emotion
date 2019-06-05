# -*- coding: utf-8 -*-
from keras.models import model_from_json
import numpy as np
import operator
list_emo = ["Angry", "Disgust","Fear", "Happy","Sad", "Surprise","Neutral"]


list_dict = dict((el,[]) for el in ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"])

class FacialExpressionModel(object):
    
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]
    
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #print("Model loaded from disk")
        #self.loaded_model.summary()

    def predict_emotion(self, img):
        list1 = []
        self.preds = self.loaded_model.predict(img)
        #print("self pred x:",self.preds)
        print("=======New Frame======")
        for i in range(len(self.EMOTIONS_LIST)):
           
            print(self.EMOTIONS_LIST[i],self.preds[0][i])
            #list_dict[self.EMOTIONS_LIST[i]].append(self.preds[0][i])
            list1.append(self.preds[0][i])
            
        #print(FacialExpressionModel.EMOTIONS_LIST[0])
        #print (list1)
        #updated_list = []
        enum_list = list(enumerate(list1))
        enum_list.sort(key=operator.itemgetter(1),reverse=1)
        if enum_list[0][0] == 6 :
            if enum_list[0][1] < 0.6: #neutral is the biggest
                list1[6] = enum_list[0][1]/2
                list1[enum_list[1][0]] = list1[enum_list[1][0]] + list1[6]
          
            
        #print (list1)    
        #return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)],list1
        return list_emo[list1.index(max(list1))],list1


if __name__ == '__main__':
    pass
