# Kpop Idol Recognition Tool
A tool that detects and recognizes k-pop idols’ faces in music videos in real-time given 10-15 input pictures using keras.

Problem Description 
===================

With BTS’s and BlackPink’s rising popularity in the last few years,
K-pop has become a global phenomenon. It has gathered millions of fans
and record-breaking achievements (e.g., BTS has more than 40 million
official fans, and their song Dynamite has topped the Billboard Artist
100 chart for 13 consecutive weeks) (Billboard, 2020). As more people
get interested in Korean pop music, the number of those who get confused
by the similarity in the looks of k-pop idols rises as well. Unless you
are an experienced k-pop stan or just really good at recognizing people,
telling idols apart, especially in MVs (Music Videos), is difficult. I
am an A.R.M.Y. (i.e., the name for BTS fans) myself and I observed that
many new fans can’t recognize all members. For example, they cannot
differentiate V and Jungkook (JK). The below image is a merge of them.
Can you tell the difference?

![pic](https://user-images.githubusercontent.com/47840436/124881430-25401080-dff1-11eb-84e3-3ce98c38da40.jpg)

I was interested in the reasons behind, so I dug deeper and found out
that there is a phenomenon called the “Cross-race identification bias.”
It says people of one race find it challenging to recognize the facial
features and expressions of people of another race. That made sense
because most of the fans are international so they are not used to
seeing Korean faces a lot. Although I am Asian and I am used to seeing
Korean faces (i.e., I became interested in Korean culture in 2013), I
sometimes find it hard to memorize the faces within the new k-pop
groups. This is also because they often have to go through plastic
surgeries to fit Korean beauty standards and look alike (Stone, 2013).

This is why it was interesting for me to try building an algorithm that
could tell the members apart and indicate their names so that it would
be helpful to fans. I will create two versions of the face recognition
algorithm: one using the pre-existing libraries, and the other one by
fine-tuning the VGGFace model created by the
<a href="http://www.robots.ox.ac.uk/~vgg">Visual Geometry Group (VGG) at Oxford</a>.

Solution Specification 
======================

Optimization 
------------

As mentioned, I aimed to use two methods: one with the existing package
that detects and recognizes the faces automatically and the other
through fine-tuning the VGGFace model. I explained them below.

Face Recognition Library 
------------------------

For this approach, I used the \`face\_recognition\` library (see
Documentation
<a href="https://buildmedia.readthedocs.org/media/pdf/face-recognition/latest/face-recognition.pdf"> here</a>)
for the face detecting and encoding steps. For processing the videos, I
used the cv2 library.

We would need to consider the following:

1.  **KNOWN FACES:** since we have some faces that need to appear in a
    particular video (i.e., BTS members in our case), the newly detected
    faces need to be compared with the known ones. To do that, we need
    to encode the known faces (i.e., return the most important features
    in a face that would differentiate that specific person), label them
    correspondingly and save them. The encoding depends on how the model
    was trained and what it thinks is crucial for the face. The
    face\_recognition library automatically does it for you (i.e.,
    face\_encodings method) based on their pre-trained (i.e., on adults)
    model.

2.  **PROCESS VIDEO:** Now, we need to process the video frame by frame
    since we would detect and recognize the faces on each picture. I
    will use cv2 to write the input video and then add rectangles around
    the faces with the corresponding members’ names. Then, the final
    frames will be written to a new file and saved on the computer.

3.  **FACE DETECTION:** The problem of face detection is understanding
    where the face is on a given frame. The face\_recongition library
    does this through the command face\_locations, which returns an
    array of boxes (coordinates) of faces in an image. It uses dlib for
    these purposes.

4.  **FACE RECOGNITION:** Based on the new face that we detected in the
    previous step, it encodes it in the same fashion as described above.
    There are different methods for encoding, from which the library
    provides:

    1.  **face\_distance:** calculates the euclidean distance as a means
        of comparison between the given face encodings and the known
        face encodings. The distance basically serves as a metric of how
        similar the faces are (i.e., the smaller the value, the closer
        they are).

    2.  **compare\_faces**: compares the given face encodings with the
        known face encodings to see if they match. Returns True/False
        based on the given tolerance, so it is similar to the
        face\_distance method but includes a tolerance.

I will combine both of them and choose the match that a) passes the
tolerance for distance and b) has minimal distance among all encodings.

The input would be the video with BTS members, while the output is the
processed video with the names. This approach is good because of:

-   **convenience**: It has all of the methods for detecting and
    recognizing the faces built-in, so it is very convenient since we
    don’t have to build models from scratch.

-   **performance**: It has been extensively trained on thousands of
    pictures (i.e., the documentation doesn’t specify how many
    explicitly) of adults, which is why it will perform better than a
    model that was trained on a few samples. Also, it requires only a
    few data points (e.g., around ten pictures for each class were
    enough for it to be able to recognize members relatively well).

This approach has some drawbacks, too, though:

-   **not flexible**: The model for extracting the features cannot be
    changed (as far as I understand), which is why there is no way to
    tweak the model (e.g., change the layers in the model) or make it
    better.

VGGFace Fine Tuning 
-------------------

We could do feature extraction, but since the above method is already
doing it, I decided to try something different. In the previous part, we
extracted the image features using a pre-existing model and then
compared these encodings to the library of known encodings. For this
step, I performed fine-tuning (i.e., unfreezing the weights and
performing a few more iterations of gradient descent) to help the
network specialize in classifying the seven members.

I didn’t change the original VGGFace architecture much, just deleting
the top and adding an extra layer to tailor it to classifying members.
Also, I used ImageDataGenerator to automatically import the images and
corresponding classes from the directories. I used a sparse categorical
cross-entropy loss function, which is the same as categorical
cross-entropy but for indexed labels (i.e., not one-hot encoded). This
is a suitable loss function because our classes are mutually exclusive.
I first compiled the model as usual (i.e., frozen weights) and then
performed a few more iterations with a small learning rate with the
weights unfrozen.

The training accuracy was 69.3% (192 pictures), and the testing accuracy
was 48.14% (27 photos), which indicates that the model was overfitting
(i.e., fit too closely on the training set). This might be because I had
very few pictures for the testing set and the neural network
architecture was not suitable. I tried to improve the performance by
adding extra Dropout layers and changing the complexity of the
architecture, but it didn’t seem to help that much. This is why I ended
up adding more data, which helps in most cases since the model will have
more instances of the members and take away more important features.
**Note**: Both the VGGFace and the Face Recognition library were trained
on the same amount of data.

The rest of the algorithm was quite similar to the previous model since
I analogously used each frame, detected faces in it, cropped the face,
and fed the picture into the neural network so that it classifies the
member.

Once again, the input would be the video with BTS members, while the
output is the classified video. This approach is good because of:

-   **flexibility**: I could change the model and the network
    architecture depending on what I need, as well as train the model on
    the data I have. There is a lot of room for improvement and
    experimentation.

-   **relevance**: The VGGFace model was pre-trained on thousands of
    faces already, so it is a good model to start the face recognition
    exploration.

However, it has its own disadvantages:

-   **performance**: It doesn’t perform as well as the existing package,
    although we use the pre-trained VGGFace model.

-   **time**: It takes more time than the face\_recognition library to
    compile.

Testing and Analysis
====================

I tested one video with labels and one video without tags. The original
videos were taken from BTS’s “Black Swan” and “Boy With Luv” music
videos. Below are some screenshots from the two models for the labeled
video (see Appendix A for code for Face Recognition library and Appendix
B for VGGFace).

<div align="center">
<img width="938" alt="pic1" src="https://user-images.githubusercontent.com/47840436/124882058-d47ce780-dff1-11eb-8309-69963ccf0627.png">
</div>

<div align="center">

Figure 2: (Correct) Face Recognition model screenshot (<a href="https://drive.google.com/file/d/1uwWMOzC-5KeguCB8JcsJjxayJsUyJM2Y/view?usp=sharing">link to video</a>)
</div>

  <div align="center">

<img width="926" alt="pic2" src="https://user-images.githubusercontent.com/47840436/124882066-d646ab00-dff1-11eb-89c5-1691e748c6d4.png">
</div>

<div align="center">
Figure 3: (Incorrect) VGGFace model screenshot (<a href="https://drive.google.com/file/d/1v0N-LGEBKLcNhBAI9_2lvhZKxh1THAli/view?usp=sharing">link to video</a>)
</div>

<div></div>
I used the same metric to assess the performance of the models. I used
the **accuracy score** (i.e., ratio of the correctly identified labels
over the total number of labels). Since I was working with videos, I
considered each frame as a test data point, so for the video I tested, I
got 479 points since it had 479 frames. I manually labeled the videos
and then checked whether the members are correctly classified. This is a
good metric as the models have never seen the frames before.

Face Recognition Library 
------------------------

I got an accuracy of approximately 80.2%, which is pretty good given
that the model didn’t know what members looked like in the music video
specifically. Also, each member had only around 25-30 photos each (a
total of 192 photos for 7 categories), so I think the performance is
relatively good.

An interesting observation about the library model is that the biggest
thing it was confused about is whether V was JK or not. As mentioned at
the beginning of the assignment, new fans find it hard to recognize
these two (i.e., it becomes relatively easy after some time).

VGGFace
-------

For the neural network model, the accuracy score isn’t as high, with
65%. This is because we had a small dataset, 7 categories, and BTS
members mostly had the same makeup and lighting, so the model confused
members a lot.

More testing 
------------

I wanted to label the second video as well using a similar technique. I
would have had labels for each frame, and if there are two or more
people, then based on which one is ordered first, I would have had
multiple labels for each frame. I would then count each label-prediction
pair as an observation so that I would have more data points. I wanted
to test the second video since the first one showed each member
separately, and they are not moving around a lot. However, the second
video had a lot more movements (i.e., mainly dancing), and it ended up
taking me a lot more time than I initially thought. This is why I
decided to keep only one video labeled (i.e., it is easier and still has
enough frames for a fair estimate) and visually inspect the second video
results.

From visual inspection, the face\_recognition library is doing well. It
misclassified only a few people and for a small period while recognizing
JK, Jimin, and J-Hope the best. The VGG16, on the other hand, is not
doing as great, constantly mixing some of the members.

The video using the Face Recognition library could be found <a href="https://drive.google.com/file/d/1GnagpRxhsXDuVFA5hiupZ7hufonhY5ft/view?usp=sharing ">here</a>, while the video using the VGGFace could be found <a href="https://drive.google.com/file/d/1XabYDhgzZsmtsfs-lvwLIm01eyi6HlT7/view?usp=sharing">here</a>. 

Conclusions
-----------

In this assignment, I tried to create a tool that would differentiate a
South-Korean boy group BTS. Both models worked, although the VGGFace one
didn’t perform as well as the face\_recognition library. In any case,
there is still room for improvement, starting from the method I chose
for classification to labeling and testing videos. We could also get
more data since thousands of k-pop idol photos are available online,
although there are no sorted, labeled datasets that could easily be
downloaded. These models could be applied for classifying other k-pop
groups as well.

I am interested in this topic, which is why I might explore it in more
detail for my Capstone. I am thinking about something like a Google
Chrome extension that would automatically apply it to videos. Although
there are many more complications (e.g., time, accuracy) that need to be
fixed, I feel like this idea is very relevant, especially given the
increasing interest in k-pop.

References 
==========

1.  Big Hit Labels. (2020). BTS (방탄소년단) ’Black Swan’ Official MV.
    [Video File]. Retrieved from
    https://www.youtube.com/watch?v=0lapF4DQPKQ

2.  Big Hit Labels. (2019). BTS (방탄소년단) ’작은 것들을 위한 시 (Boy
    With Luv) (feat. Halsey)’ Official MV. [Video File]. Retrieved from
    https://www.youtube.com/watch?v=XsX3ATc3FbA

3.  Billboard. (2020). BTS Becomes First Group to Rule Artist 100, Hot
    100 & Billboard 200 Charts at the Same Time. Retrieved from
    https://www.billboard.com/articles/business/chart-beat/9492020/bts-first-group-rule-artist-100-hot-100-billboard-200-charts/\#: :text=BTS

4.  Geitgey, A. (2020). Face Recognition Documentation. Retrieved from
    https://buildmedia.readthedocs.org/media/pdf/face-recognition/latest/face-recognition.pdf

5.  Stone, Z. (2013). The K-Pop Plastic Surgery Obsession. Retrieved
    from
    https://www.theatlantic.com/health/archive/2013/05/the-k-pop-plastic-surgery-obsession/276215/)

