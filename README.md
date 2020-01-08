# image-sentiment-analysis-ml
machine learning project applying sentiment analysis to images

front end: angular
back  end: django rest framework

machine learning: neuronal nets with backpropagation (as explained in Andrew Ng's Course Introduction to Machine Learning)

Process:
Upload Image (or Take Selfie)
Image is processed to Gray Scal and Right Dimensions
Send To Server
Server Applies trained Machine Learning Model to processed Image (Model has been trained already)
Return Result




Current Status: Model needs to be adapted and trained before app is usable


open decisions: file upload to server, aws s3 or not at all?
image manipulating to correct dimensions on client side or server side? (currently server side, the image manipulation in python is more convenient)
