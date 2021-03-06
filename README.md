# Comments Data Analysis using NLP and Machine Learning

Api Link : https://secret-peak-13498.herokuapp.com/

`Modified Version` : https://github.com/Gaurav0824/Api-majorProject-server

`Modified Api Link` : https://protected-gorge-33386.herokuapp.com/

<br>

This is the backend api serving the <a href = "#website">website</a>

This api is used to perform the following tasks:

- Spam Filtering (Naive Bayes)
- Sentimental Analysis (Vader)
- Topic Modelling (LDA)

        Topic modelling is computationally expensive and does not work with free tier accounts.
        The app dyno kills on high cpu usage.
        The api needs to be hosted on a paid account to use topic modelling.
        A work-around is to use LDA available with nodejs and perform topic modelling on the client side.
           https://www.npmjs.com/package/lda

- Create Word-Clouds

<hr>

<div id="website"></div>
        
:computer: **WebSite** built using this api : https://fastidious-taffy-09c112.netlify.app/ 

<hr>

:electron: **Github** Link : https://github.com/Gaurav0824/MajorProjectWebsite

<hr>

:orange_book: **Colab** Link : 

- Spam : https://colab.research.google.com/drive/11-ZTT6o9waxcJ3wmNUEWyBNlalRs-gIq?usp=sharing
- Sentiments : https://colab.research.google.com/drive/1xW4B06L3B1TEqLOKPkx19o8Zzg8nwqkA?usp=sharing

<hr>

## ScreenShots

<br>

<details><summary>Spam  </summary>

<br>

https://secret-peak-13498.herokuapp.com/spam

<br>

<!-- ![](img/2022-06-22-17-35-27.png) -->

![](img/2022-06-22-17-48-32.png)

<br>

</details>

<details><summary>Sentiment  </summary>

<br>

https://secret-peak-13498.herokuapp.com/sentiment

<br>

<!-- ![](img/2022-06-22-17-47-05.png) -->

![](img/2022-06-22-17-49-12.png)

<br>

</details>

<details><summary>Word Cloud  </summary>

<br>

https://secret-peak-13498.herokuapp.com/wordcloud

<br>

| Params    | Description                      |
| --------- | -------------------------------- |
| word      | Concatenated String of sentences |
| max_words | Max. number of words to include  |
| width     | width in pixels                  |
| height    | height in pixels                 |

<br>

![](img/2022-06-22-17-51-06.png)

<hr>


    Use this encoded output in image tag as `base64` string to create a word cloud.

    <img src="data:image/png;base64, {{ word_cloud_encoded_string_from_api }}" alt="word cloud">

![](img/2022-06-22-18-22-08.png)

![](img/2022-06-22-18-18-44.png)

</details>
