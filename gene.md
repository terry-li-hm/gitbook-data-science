# Gene

{% embed data="{\"url\":\"https://buhrmann.github.io/tfidf-analysis.html\",\"type\":\"link\",\"title\":\"Analyzing tf-idf results in scikit-learn - datawerk\",\"icon\":{\"type\":\"icon\",\"url\":\"https://buhrmann.github.io/theme/css/logo.png\",\"aspectRatio\":0}}" %}

 Tf-idf stands for_ term frequency-inverse document frequency_

> Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency \(TF\), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency \(IDF\), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
>
> * **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length \(aka. the total number of terms in the document\) as a way of normalization:   TF\(t\) = \(Number of times term t appears in a document\) / \(Total number of terms in the document\).
> * **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:   IDF\(t\) = log\_e\(Total number of documents / Number of documents with term t in it\).
>
> See below for a simple example.
>
> **Example:**
>
> Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency \(i.e., tf\) for cat is then \(3 / 100\) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency \(i.e., idf\) is calculated as log\(10,000,000 / 1,000\) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 \* 4 = 0.12.

