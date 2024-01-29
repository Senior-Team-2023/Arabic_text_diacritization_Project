# Arabic Diacritic Classifier

## Introduction

This project tackles the challenging problem of Arabic text diacritization using an RNN approach. Through extensive experimentation, we explored various models, with the most effective ones identified as the "BaseLine Model" and "CBHG."

Here is an example of our Arabic text diacritization:

<p align="center"><code>ذَهَبَ عَلِيٌ إلَى الْشَّاطِئِ → ذهب علي إلى الشاطئ</code></p>

## Models Architecture
![baseline](https://github.com/markyasser/Arabic-Diacritic-Classifier/assets/82395903/1ca08ad9-d3c1-4289-b413-8415f3ff4980)

BaseLine Model
<hr>

![cbow model](https://github.com/markyasser/Arabic-Diacritic-Classifier/assets/82395903/dcd80b4d-d39f-4b89-b0cc-317162b0ab05)
![cbow](https://github.com/markyasser/Arabic-Diacritic-Classifier/assets/82395903/e6453f25-84fd-4d00-9081-eb4c8294021f)

CBHG

## Quick start


1. Clone the repo on your local machine.
2. Install the required libraries specified in `requirement.txt` (or simply upload the files on google drive and run them using google colab).
3. To train the model, run one of the notebooks inside `Training` folder.
4. To Evaluate the model, run the notebook corresponding to the model to be evaluated inside `Evaluation` folder.
5. `BaseLine Demo` folder contains a read to use notebook that takes an arabic string from the user and diacritise it.

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
