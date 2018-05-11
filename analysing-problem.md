# Problem definition

[Facebook’s Field Guide to Machine Learning video series – Facebook Research](https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/)

[Qualitative before Quantitative: How Qualitative Methods Support Better Data Science](https://medium.com/indeed-data-science/qualitative-before-quantitative-how-qualitative-methods-support-better-data-science-d2b01d0c4e64)

> 機器學習是解決問題的方法一種，但首先必須要理解你的問題本身  
> 如果你要預測航班到點時間，你要怎麼做呢？  
> 1.幫助航空公司明白哪些航班會誤點，他們可以迅速解決問題  
> 2.幫助人們買到不容易誤點的航班  
> 3.提醒人們明天航班可能會誤點  
> 很多時候在項目裡，你對問題本身都一團霧水的時候怎麼能夠希望一個有效的模型能解決問題呢？  
> 理解問題還能幫助你做好如下決策  
> 1.我的模型的預測精確度需要到多少？  
> 2.什麼樣的假陽性是可接受的？  
> 3.我能用什麼數據？如果要預測明天的航班看天氣數據就行，如果要預測一個月以後的航班呢？
>
>  找出優化指標  
> 還是拿出航班延誤的例子。 首先我們要決定是用分類還是用回歸？拍個腦袋，用回歸吧。  
> 人們通常會優化平方和因為它良好的統計特性，但是在我們的問題裡，航班延誤10小時和20小時一樣糟糕，所以平方和合適麼？
>
>  決定用什麼數據  
> 比方說我已經有了航空公司的航班號，起飛機場，飛機型號，起飛抵達時間  
> 我還需要買別的信息麼？更精準的天氣信息？也就是說可用的信息並非固定的,只要你想就可以更多
>
>  清理數據  
> 一旦有了數據，你的數據可不是齊整的，如機場命名不一致，沒有延誤信息，奇怪的數據格式，天氣數據與機場地理位置不一致  
> 把數據整理到可用，不是件容易的事。如果還需要整合很多數據源，那麼它可能會佔你80%的時間。
>
>  建個模型  
> 這才到了Kaggle部分， 訓練，交叉驗證，Ya  
> 現在你建了一個碉堡的模型，我們要用它。



