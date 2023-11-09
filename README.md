# NLP-Project
## Project Description
Arabic is one of the most spoken languages around the globe. Although the use of
Arabic increased on the Internet, the Arabic NLP community is lagging compared to
other languages. One of the aspects that differentiate Arabic is diacritics. Diacritics are
short vowels with a constant length that are spoken but usually omitted from Arabic text
as Arabic speakers usually can infer it easily. The same word in the Arabic language
can have different meanings and different pronunciations based on how it is diacritized.
Getting back these diacritics in the text is very useful in many NLP systems like Text To
Speech (TTS) systems and machine translation as diacritics removes ambiguity in both
pronunciation and meaning. Here is an example of Arabic text diacritization: 
<p align="center"><code>ذَهَبَ عَلِيٌ إلَى الْشَّاطِئِ → ذهب علي إلى الشاطئ</code></p>

## Dataset Description
The dataset contains discretized Arabic sentences. Each sentence occupies a line. The
dataset is divided into three different portions (train, dev, and test). Both the train and
dev sets will be annotated (All characters are diacritized). The test set contains Arabic
text without diacritization and your task is to restore the test set diacritics. The dataset
portion sizes are as follows:
1. The training set contains 50k lines.
2. The dev set contains 2.5k lines.
3. The test set contains 2.5k lines.

## Project Pipeline
Your task is to build a system that takes a sentence and produces the same sentence
after restoring the missing diacritics. There are several approaches to tackle such a
problem. You are free to propose your own pipeline based on your understanding and
research of the problem. Here is an example pipeline diagram that you may follow:

![image](https://github.com/markyasser/NLP-Project/assets/82395903/717edc5d-ad43-4f6e-a016-087b3998f935)


The above diagram is based on the problem understanding. To restore the diacritics of a
word, we need to read several words before it and maybe after it also. So on the word
level, you need to have the features of it besides the features of the words preceding
and following it if available. To predict the diacritic of each character, this can be viewed
as a classification problem per character and the classes are the diacritics. So, for the
character level, each character needs to know the features also of the characters
preceding and maybe following it in the same word. The features then can be fed to the
final classifier. Following are some of the main steps you need to include in your project:
### 1. Data Preprocessing:
a. Data cleaning: removing undesired words such as HTML tags, English
letters, etc.
b. Tokenization: You are free to determine the best tokenization approach for
the problem.
### 2. Feature Extraction: 
you are required to try at least three different features (e.g.
Bag of Words, TF-IDF, Word Embeddings, Trainable embeddings etc.) It would
also be great if you tried other features (e.g. contextual embeddings). This part
can be done in the word and character levels.
### 3. Model Building: 
In this phase, you are required to build at least two machine
learning models (e.g. RNN, LSTM, CRF, HMM, …). Optimizing the model weights
will be done using the training set. You will need to use the dev set to pick the
model that performs the best. You can have multiple models in your system
based on your approach. For example, you can have a model for word-level
encoding and another for character level.
### 4. Model Testing: 
In this phase, you will use your best-performing model to
produce the stance and category of the given test set.



## Collaborators
<table>
<tr>
    <td align="center">
        <a href="https://github.com/markyasser">
            <img src="https://avatars.githubusercontent.com/u/82395903?v=4" width="150;" alt="Mark Yasser"/>
            <br />
            <sub><b>Mark Yasser</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/EngPeterAtef">
            <img src="https://avatars.githubusercontent.com/u/75852529?v=4" width="150;" alt="Peter Atef"/>
            <br />
            <sub><b>Peter Atef</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/bemoierian">
            <img src="https://avatars.githubusercontent.com/u/72103362?v=4" width="150;" alt="Bemoi Erian"/>
            <br />
            <sub><b>Bemoi Erian</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/doaa281">
            <img src="https://avatars.githubusercontent.com/u/65799105?v=4" width="150;" alt="Doaa Ashraf"/>
            <br />
            <sub><b>Doaa Ashraf</b></sub>
        </a>
    </td>
  </tr>
</table>

