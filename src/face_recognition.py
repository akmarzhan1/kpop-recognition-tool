#importing the needed libraries 
import face_recognition 
import cv2 
import os 
import numpy as np 


def encoding(face_dir):
    """
    Function for encoding the 'known' faces to then store them and use for comparing.
    
    Input:
        face_dir: a directory of known faces with each person under their own folder.
    
    Output: 
        faces: list with the encoded faces (i.e., noted important features and etc. 
               from the provided sample).
        names: list with the names of the people who are encoded (i.e., corresponds 
               to the 'faces' list).
    """
    
    #initializing
    faces, names = [], []
    
    #looping through each person's folder 
    for name in os.listdir(face_dir):
        
        #os.listdir finds some documents that we don't need, so this is the easiest way 
        #to ignore them
        if name == '.DS_Store':
            continue 
        for file in os.listdir(f'{face_dir}/{name}'):
            if file == '.DS_Store':
                continue
                
            #loading images (one person at a time)
            image = face_recognition.load_image_file(f'{face_dir}/{name}/{file}')
            
            #some of the pictures I found weren't pictures but rather .webp, and I didn't
            #want to go through all of them again so I used the below line
            if len(face_recognition.face_encodings(image))!=0:
                
                #actual encodings
                encoding = face_recognition.face_encodings(image)[0]
                
                #storing the encodings of the faces and the corresponding names
                faces.append(encoding)
                names.append(name)
    #output            
    return faces, names

def in_out(inp, out):
    """
    Function for reading the input video and initializing the output video.
    
    Input:
        inp: path to the input video that we want to process.
    
    Output: 
        input_mov: VideoCapture object containing the input video file.
        output_mov: VideoWriter object which will store the output video.
    """
    
    #setting the input video we want to process
    input_mov = cv2.VideoCapture(inp)

    #setting the output video to match the size of the input video and have a
    #specific name and format 
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_mov = cv2.VideoWriter(out, fourcc, 20.0, (int(input_mov.get(3)),int(input_mov.get(4))))
    
    #output
    return input_mov, output_mov


def clf(inp, out, tol, model, test, prev_acc=None):
    """
    Function for reading the input video and initializing the output video.
    
    Input:
        inp: path to the input video that we want to process.
        out: path to the output video.
        tol: path to the input video that we want to process.
        model: the model to use for face detection. 
        test: the test set to compare results. 
        prev_acc (default = None): previous accuracy list to 
                                   keep track of results.
        
    Output:
        accuracy: the list with 0/1 depending on whether the 
        classification is correct. 
    """
    
    input_mov, output_mov = in_out(inp, out)
    
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
        
        #finding all of the faces (+their encodings) in the current frame
        #for face_locations, I could either choose default "hog" or 'cnn'
        #"cnn" classifies better but it is CUDA accelerated and my computer
        #doesn't support that (i.e., after the last Catalina update)
        face_locs = face_recognition.face_locations(frame, model=model)
        face_encs = face_recognition.face_encodings(frame, face_locs)

        #repeating for each face we find 
        for idx, (face_enc, face_loc) in enumerate(zip(face_encs, face_locs)):

            #comparing encodings (i.e., known faces with candidate encodings)
            #returns True if they match, and False if they don't
            results = face_recognition.compare_faces(faces, face_enc, tol)
            
            #additional measure which compares the candidate face with the known 
            #faces and returns the euclidean distance between them (i.e., the distance
            #defines how similar the faces are - smaller is better)
            distance = face_recognition.face_distance(faces, face_enc)
            
            #find the smallest distance face
            best = np.argmin(distance)

            #check if the smallest distance face (i.e., from known) passes the tolerance
            if results[best]:
                match = names[best]
                prev = match
            else:
                #don't put a square if no face passes the tolerance (i.e., unknown face or
                #not sure enough)
                continue
                
            if match==test[int(frame_number)]:
                accuracy.append(1)
            else:
                accuracy.append(0)

            #creating squares to highlight the faces 
            top_left = (face_loc[3]-10, face_loc[0]-10)
            bottom_right = (face_loc[1]+10, face_loc[2]+10)

            #choosing white to be the color 
            color = [255, 255, 255]

            cv2.rectangle(frame, top_left, bottom_right, color, 1)

            #defining the blocks for text
            top_left = (face_loc[3], face_loc[2]+10)
            bottom_right = (face_loc[1]+10, face_loc[2] + 32)

            cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
   
            #defining the text
            cv2.putText(frame, match, (face_loc[3] + 5, face_loc[2] + 25), 
                        cv2.FONT_ITALIC, 0.45, (0, 0, 0), 1)

        #writing the processed frames to the output file
        output_mov.write(frame)
        
    #processing and cleaning 
    input_mov.release()
    output_mov.release()
    
    return accuracy 


#initialization
faces, names = encoding("faces")



bs_test = []

#the correct classifications for the corresponding frames 
#there are overall 479 frames and I labeled the frames manually
for i in range(41):
    bs_test.append("Suga")
    
for i in range(95):
    bs_test.append("V")
    
for i in range(54):
    bs_test.append("JK")
    
for i in range(50):
    bs_test.append("RM")

for i in range(93):
    bs_test.append("J-Hope")
    
for i in range(145):
    bs_test.append("Jin")



#the process itself which returns the accuracy
accuracy = clf("bs.mov", "bs_face.mov", 0.5, 'cnn', test=bs_test)


 


print("Testing accuracy (479 frames):", np.mean(accuracy))

