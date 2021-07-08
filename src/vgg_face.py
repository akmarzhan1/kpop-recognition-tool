

#importing the needed libraries
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD


 


#loading the original VGGFace model without the top
vgg = VGGFace(include_top=False, input_shape=(224, 224, 3))

#setting the few layers as non-trainable because they usually capture
#the universal features that might be needed for our task too
for layer in vgg.layers:
    layer.trainable = False
    
last_layer = vgg.output

#building new layers for preventing overfitting
x = Flatten(name='flatten')(last_layer)
out = Dense(7, activation='softmax')(x)

model = Model(vgg.input, out)


 


from keras.preprocessing.image import ImageDataGenerator

#preprocessing the images
batch_size = 4
nrow, ncol = 224, 224
epo = 5

#flowing all the images from the training directory
train_data_dir = './faces'
tr_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2)
train_generator = tr_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='sparse')


 


#adding some arbitrary images to see how well the model
#can differentiate the members
test_data_dir = './test_faces'
te_datagen = ImageDataGenerator(rescale=1./255)
test_generator = te_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='sparse')


 


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.fit_generator(train_generator, steps_per_epoch=20,
    epochs=epo, validation_data=test_generator)


 


#setting the next layers as trainable 
for layer in model.layers:
    layer.trainable = True
    
#performing a few more iterations of gradient descent with a small learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-7), metrics='accuracy')
model.fit_generator(train_generator, steps_per_epoch=10, epochs=epo, validation_data=test_generator)


 


model.evaluate(test_generator)


 


import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

def matching(frame, results, model, test, frame_number):
    """
    Function for reading the input video and initializing the output video.
    
    Input:
        frame: the current frame we need to process. 
        results: the output from the face detection package.
        model: the classification model we are using. 
        test: the test labels.
        frame_number: the current frame number.
    
    Output:
        frame: processed frame. 
    """
    
    #going through all of the identified faces (i.e., locations)
    for result in results:
        x, y, w, h = result['box']
        #cropping out the faces so that it is easier to classify
        crop = frame[y:y+h, x:x+w]
        
        #preprocessing the frames so they would fit into the classifier
        output = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        output = cv2.resize(crop, (224, 224)).astype("float32")
        #### try with mean
        
        ####### try without [0]
        #predicting the class (i.e., gives probabilities for all seven
        #so we should extract the highest probability)
        prediction = model.predict(np.expand_dims(output, axis=0))[0]
        y_pred = prediction.argmax(axis=-1)
            
        #finding the name of the match 
        match = list(labels.keys())[list(labels.values()).index(y_pred)]
        
        #drawing the rectangles
        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10),
                  (255, 255, 255), 1)
        cv2.rectangle(frame, (x-10, y+h+10), (x+w+10, y+h+30), 
                      (255, 255, 255), cv2.FILLED)

        cv2.putText(frame, match, (x + 5, y+h+25), 
                cv2.FONT_ITALIC, 0.45, (0, 0, 0), 1)

        #testing 
        if match==test[int(frame_number)]:
            accuracy.append(1)
        else:
            accuracy.append(0)
        
    return frame, accuracy


 


from mtcnn import MTCNN
def vgg_clf(inp, out, model, test=None, prev_acc=None):
    """
    Function for reading the input video and initializing the output video.
    
    Input:
        inp: path to the input video that we want to process.
        tol: path to the input video that we want to process.
        model: the NN model to use for classification.
    
    Output:
        accuracy: the list with 0/1 depending on whether the 
        classification is correct. 
    """
    
    input_mov, output_mov = in_out(inp, out)
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

    #seeing whether we should keep track of some previous accuracy
    if prev_acc:
        accuracy = prev_acc
    else:
        accuracy = []
        
    while True:
        
        #reading each frame 
        ret, frame = input_mov.read()
        frame_number = input_mov.get(1)
        
        #breaking when the video is over
        if not ret:
            break
        
        #detecting faces and classifying the people
        detector = MTCNN()
        results = detector.detect_faces(frame)
        new_frame, accuracy = matching(frame, results, model, test, frame_number)

        #writing the processed frames to the output file
        output_mov.write(new_frame)
        
    #processing and cleaning 
    input_mov.release()
    output_mov.release()
    
    return accuracy 


 


labels = train_generator.class_indices
    
#the process itself which returns the accuracy
accuracy = vgg_clf('bs.mov', "bs_vgg.mov", model, test=bs_test)


 


print("Testing accuracy (479 frames):", np.mean(accuracy))

