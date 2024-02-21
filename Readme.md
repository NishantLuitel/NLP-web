# Nepali Language Processing

## Abstract
This project is built in order to explore the natural language processing field and understand the working of transformers and their uses to develop various projects like language models, text summarization, text classification and more. We have developed sentimental classification model, word vectors for nepali words, probabilistic language model and GPT2-based language model for nepali language in this project and the work is on going.


## Project Goals

* To develop nepali language model using probabilistic and sequential model.
* Explore the areas of word embeddings and classification of nepali texts.
* Develop a spelling correction model of nepali texts

## Frontend and Backend
Frontent is developed using React and Backend is developed using django and django rest framework.

* [Backend](https://github.com/NishantLuitel/NLP-web)
* [Frontend](https://github.com/AAreLaa/NLP-UI)

## Status 
* Currently working on.

## Steps to run locally
**Clone the repository**
```
git clone https://github.com/NishantLuitel/NLP-web
```

**Create a virtual environment and activate it**
```
virtualenv venv
source ./venv/bin/activate
```
**Download the Text Generation and Spelling Correction Model**
After the downloading the models from following drive, put it into the **NLP_Trained_models** folder
```
https://drive.google.com/drive/folders/1BxDi60220mTRzT7pA3owfrTz7EoB1dqJ?usp=sharing
```

**Install all requirements from requirements.txt**
```
pip install -r requirements.txt
```

**Create a superuser**
```
python3 manage.py createsuperuser
```

**Run Server**
```
python3 manage.py runserver
```


