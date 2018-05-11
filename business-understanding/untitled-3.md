# Use Cases

## Banking

* 风险监控
* 智能投顾 \(seems not clear evaluation metric?\)
  * 通过大数据获得用户个性化的风险偏好及其变化规律
  * 根据用户个性化的风险偏好结合算法模型定制个性化的资产配置方案
  * 利用互联网对用户个性化的资产配置方案进行实时跟踪调整
  * 不追求不顾风险的高收益，在用户可以承受的风险范围内实现收益最大化
  * generating targeted trading strategies to share with clients: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf).
* Credit Risk Management
  * 传统银行在贷款的时候会对贷款主体进行风险识别和控制，避免把钱带给信誉较差的带块主体导致坏账。当然传统银行更多的是靠人和流程来控制，P2P在这块的创新主要是运用了计算机算法，输入贷款主体的特征（如收入、职业、历史还款记录等），通过逻辑回归或者机器学习的方法，算出贷款主体未来按期还款的概率，提升了单一靠人贷款工作的效率。
  * According to [Accenture](http://financeandriskblog.accenture.com/analytics/where-can-machine-learning-be-applied-to-improve-banking-performance): _The current credit risk workflow tends to be labour intensive, slow and riddled with judgement related human-errors. Machine Learning credit default prediction models allow for more accurate, instant credit decisions as they can automatically use a much broader range of data sources including news and business networks. The algorithms can also be used to improve Early Warning Systems \(EWS\) and to provide mitigation recommendations, based on previous responses. The result is lower rates of default losses whilst also reducing the risk of losing customers to competitors due to a slow process._
* 身份認證: 將活體識別、聲紋識別等技術用於身份識別，可運用於遠程開戶、遠程授信、刷臉支付等金融交易環節和場景。並且已經和泰康人壽進行合作，進行投保業務的身份認證，也應用在了自己的金融產品百度錢包中，還包括空港易行的機場 VIP 身份認證。
* 業務智能分析: 摩根大通在今年 1 月份上線了 X-Connect 用來檢索電子郵件，幫助員工找到與潛在客戶關係最密切的同事，促進業務機會。
* Investment Management: As demonstrated in J.P. Morgan's Guide to Big Data and AI Strategies
* Recommend follow-on equity offerings: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf), in 2015 JPMorgan introduced "the Emerging Opportunities Engine, which helps identify clients best positioned for follow-on equity offerings through automated analysis of current financial positions, market conditions and historical data."
* Extract important data points and clauses from legal documents: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf): "As an example, we recently introduced COiN, a contract intelligence platform that uses unsupervised machine learning to analyze legal documents and to extract important data points and clauses. In an initial implementation of this technology, we can extract 150 relevant attributes from 12,000 annual commercial credit agreements in seconds compared with as many as 360,000 hours per year under manual review. This capability has far-reaching implications considering that approximately 80% of loan servicing errors today are due to contract interpretation errors." But not sure how the analysis results can be further used.
* Fraud Detection
  * detecting anomalies for fraud: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf).
  * According to [Accenture](http://financeandriskblog.accenture.com/analytics/where-can-machine-learning-be-applied-to-improve-banking-performance): _Traditional fraud systems identify fraudulent transactions based on specified, non-personalised rules, such as if a customer spends money abroad. Machine Learning systems, on the other hand, analyse large amounts of each customer’s transactions to understand their personal spending patterns. This way, they can spot subtle anomalies that indicate potential fraud. Each transaction is automatically analysed in real-time, and is given a fraud score which represents the probability that it is fraudulent. If it is above a certain threshold, a rejection is triggered immediately. This would be extremely difficult without Machine Learning techniques– as a human could not review thousands of data points in milliseconds and make a decision._
* 中小企業融資: 之所以中小微企業融資難，核心原因在於中小微企業融資渠道依賴銀行，而銀行門檻高、標準高，正如馬雲提出「新金融」時所言：傳統銀行是為有錢人服務的，因為20%的頂部客戶帶來80%的利潤。銀行因為風控要求嚴格，貸款給中小微企業會相當謹慎。對於中小企業這個群體，互聯網公司早已瞄準，阿里巴巴網商銀行主營業務之一是企業貸款，百度則向百度聯盟生態的站長和開發者推出了「聯盟貸」，但這些貸款服務均是B2C模式，只面向特定領域，比如商家、站長、供應鏈金融，能夠覆蓋的中小企業數量有限。民間具有資金能力的銀行和小貸公司還有很多，如何連接這些金融機構和中小企業？答案就是B2B金融平台。隨著科技金融的發展，只要有一定徵信的個人在手機上點點菜單就可以快速貸款甚至秒批，BAT都在搶著給你我借錢，為什麼不將這種服務也提供給中小企業呢？人們在手機上更容易貸款，是因為互聯網金融平台通過大數據做好了風控。同樣，只要能夠通過大數據做好中小企業風控，那麼金融機構就不可以抓住龐大的中小企業客群，這就是「貴州金融大腦」的思路：通過融合貴州省政務、企業、金融、互聯網等多渠道數據，再借助於人工智能技術進行智能風控，讓金融機構可以有效、快速、低成本地評估中小企業的信用，做出貸款決策。在放貸之後則採集數據進行貸後風險預測和監控，及時發現企業經營異常——這相當於傳統銀行的回訪。大數據風控技術早已被應用在百度消費金融業務上，通過智能身份識別、智能風控諸多技術，百度金融教育信貸業務甚至可以做到「秒批」。因此，「貴州金融大腦」不只是可以解決中小企業融資難、融資貴的問題，還可避免繁雜的手續、大量的調查、冗長的流程，智能審批、快速放款，畢竟對於生存能力薄弱、抗風險能力差的中小企業來說，有時候資金鏈斷裂，很可能就倒閉了。對於金融監管機構而言，中小企業和金融機構的金融借貸行為在互聯網上進行，可追溯、可監控、可預警，因而能實現更好的金融管理。另一個角度來看，金融監管機構可以推動政務大數據開放，比如工商資料、納稅記錄、司法記錄等等，來幫助金融大腦做好大數據風控。總之，「金融大腦」越普及，政府金融監管就越有效率，這是貴州省政府金融辦參與建設「貴州金融大腦」 的原因之一。\([http://column.iresearch.cn/b/201705/800281.shtml](http://column.iresearch.cn/b/201705/800281.shtml)\)
* Blockchain
  * Challenge: Crypto-currency and distributed ledger technology are among the most exciting developments in fintech. The ability to record transactions in a transparent manner without an intermediary has revolutionary applications for the financial ecosystem. But maintaining security and trust in the system is paramount.
  * Solution: DataRobot’s automated machine learning platform is well-suited to identify and help prevent identity theft, fraud, and illicit transactions in the blockchain, by developing and deploying algorithms that can detect anomalous behavior anywhere along the chain.
* Credit Card Fraudulent Transactions: DataRobot predicts which credit card transactions are most likely fraudulent based on transaction characteristics.
  * Challenge: Fraudulent transactions are costly, but it is too expensive and inefficient to investigate every transaction for fraud. Even if possible, investigating innocent customers could prove to be a very poor customer experience, leading some clients to leave the business.
  * Solution: Using DataRobot, you can automatically build extremely accurate predictive models to identify and prioritize likely fraudulent activity. Fraud units can then create a data-based queue, investigating only those incidents likely to require it. The resulting benefits are two-fold. First, your resources are deployed where you will see the greatest return on your investigative investment. Additionally, you optimize customer satisfaction by protecting their accounts and not challenging innocent transactions.
* Credit Default Rates: DataRobot operationalizes real-time loan assessment by automating model building to predict the likelihood of future default.
  * Challenge: People with years of experience can judge someone’s likelihood of default, but that is not an efficient method for judgement as you grow a business. Increasing the value and number of loans is difficult without a scalable method in place to judge the default risk of many applicants.
  * Solution: DataRobot uses past information about borrowers’ default rates to predict the likelihood of default for future borrowers. Incorporating the predictive models built with DataRobot into a real-time loan approval process is easy, allowing businesses to scale up and expand their loan portfolios.
* Risk and Finance Reporting
  * According to [Accenture](http://financeandriskblog.accenture.com/analytics/where-can-machine-learning-be-applied-to-improve-banking-performance): _Cognitive automation appears to be the next development in the world of automation, post Robotic Process Automation \(RPA\). RPA allows a business to map out simple, rule-based processes and have a computer carry them out on their behalf. Cognitive automation, however, combines this with the ‘thinking’ work of Machine Learning, and programs computers to read and understand unstructured data or text and make subjective decisions in response, similar to a human. This has the potential to transform back office risk and finance reporting processes, enabling banks to meet regulatory reporting requirements at speed, whilst reducing costs._
* Trading Floors
  * According to [Accenture](http://financeandriskblog.accenture.com/analytics/where-can-machine-learning-be-applied-to-improve-banking-performance): _With an unstable economic environment creating new risks, with profit margins trending down and continued high regulatory pressure, banks should be exploring the power of Machine Learning across their operations. The benefits include lower cost bases and improvements in the effectiveness and efficiency of processes, whilst also providing a better level of service and enhanced products and services to their customers._
* money laundering
* Identity Mind Global: Provides anti-fraud and risk management services for digital transactions by tracking payment entities
* Trunomi: Securely manages the consent to use customer personal data
* Suade: Helps banks to submit required regulatory reports without disruption to the banks’ architecture
* Silverfinch: Connects asset managers and insurers through a fund data utility to meet Solvency II requirements
* Passfort: Automates the collection and storage of customer due diligence data.
* Fund Recs: Oversees how data is managed and processed by the fund industry
* Real-Time Compliance
* 加州Mountain View有一个Google Venture投资的科技公司叫Orbital Insight。他们专门用神经网络和深度学习去识别商业卫星拍摄的图片（我猜测大部分是和object recognition相关的技术），然后推测出很多和经济/金融相关的数据/信号。比如他们用卫星拍摄的沃尔玛门前的停车场图片，查一查汽车的数量，去推测这个季度沃尔玛的销售情况
* 拍摄炼油厂和储油罐的照片进行分析，来推测现在市场上的石油供应量
* 答主在某咨询公司做AI方向的咨询，客户包括各种金融服务类公司。比较常见的案例包括Anti-money Laundering Transactions\(反洗钱\)，Smooth M&A Model \(用于提高并购质量的AI 模型\)。虽然真正的交易模型一般都是金融公司独立开发的，但还是有机会和客户讨论金融AI的趋势和模型。
* [How Artificial Intelligence Is Disrupting Finance](https://www.toptal.com/finance/market-research-analysts/artificial-intelligence-in-finance)
* 在用户获取方面，定制化的产品能够提高用户的参与度，而对机器学习的投资也是应对来自金融科技公司的竞争的办法。
* 在合规方面，机器学习能够帮助进行反欺诈、反洗钱调查并协助进行风险评估，同时，也可以通过自动化的压力测试和行为分析来监测可疑的公司内部行为。
* 在经营效率方面，自动化的操作能够通过降低人工错误率来节约成本，而自然语言分析等技术也可以帮助 HR 更精确地获取所需人才。
* 金融机构和销售商正在使用人工智能和机器学习方法来评估信用质量、为保险合约定价并进行营销、自动化客户交流过程。
* 机构正在利用人工智能和机器学习技术优化稀缺资本的分配、调整回溯测试模型，并分析大额交易的市场影响。
* 对冲基金、券商和其他公司正在使用人工智能和机器学习来寻找高收益（且没有相关性），并优化交易执行。
* 公共部门和私营机构都可以使用这些技术进行合规监管、监察、数据质量评估和欺诈检测。
* 使用人工智能和机器学习的优先级如下：以客户为中心优化流程；增加系统与员工之间的互动并加强决策能力；开发提供给客户的新产品与新服务。
* 新的合规要求也产生了一些需求。新合规要求增加了对效率的要求，这促使银行推动自动化进程并应用新的分析工具，包括含有人工智能和机器学习的工具。金融机构正在寻求遵循审慎性监管、数据报告、交易执行优化以及反洗钱和打击资助恐怖主义（AML/CFT）等监管要求的高效手段。相应地，监管机构也面临着评估更大、更复杂、增长更迅速的数据集的责任，需要用更强大的分析工具来更好地监控金融部门。

Read more: Regtech Definition \| Investopedia [http://www.investopedia.com/terms/r/regtech.asp\#ixzz4qIpvWuuN](http://www.investopedia.com/terms/r/regtech.asp#ixzz4qIpvWuuN)Follow us: Investopedia on Facebook

今年一季度，摩根大通开始在欧洲股票算法业务中投入使用 AI 应用 LOXM，旨在以最佳价格和最快速度执行客户交易指令。近日，由于 LOXM 的优异表现，摩根大通计划在四季度将其运用扩大至亚洲及美国地区。LOXM 并非只是机械地执行任务，它能够利用「深层强化学习」方法，从过往的数十亿条实盘和模拟盘的历史交易中进行学习总结，归纳经验和教训，以解决更加复杂的问题，比如怎样才能在不惊扰市场价格的情况下大量抛售股票。但与一些私人银行提供的机器人投资顾问不同，LOXM 没有做决策的能力，无法决定买卖标的，它的作用仅仅是如何买入卖出。LOXM 未来发展的一个方向是识别特定的用户，从他们的言行举止中决定如何交易。「但这些只有在客户同意的情况下才会进行。」摩根大通全球股票电子交易负责人 Daniel Ciment 补充道。

摩根大通全球股票電子交易負責人Daniel Ciment對英媒表示，從1季度起已在歐洲股票算法業務部門採用AI機器人LOXM執行交易，並計畫在4季度將其運用擴大至亞洲及美國地區。據介紹，LOXM的工作就是用最快的速度、以最優價格來執行指令。除了執行交易指令，人工智能還有很多其他的潛在用途，比如自動進行對沖操作或者做市。LOXM能夠利用「深層強化學習」方法，從過去幾十億條實盤和模擬盤的歷史交易中進行學習總結，歸納經驗和教訓，以解決更加複雜的問題，比如怎樣才能在不驚擾市場價格的情況下大量拋售股票。不過，摩根大通的LOXM沒有做決策的能力，它的作用僅僅是如何買入賣出。另外，瑞銀最近亦部署了人工智能來處理客戶的盤後交易分配申請（Post Trade Allocations）。今年3月，全球最大資產管理公司貝萊德亦炒掉7個基金經理，轉而用計算機算法作出的量化投資策略來取代。美國金融博客ZeroHedge認為，一旦貝萊德這樣的巨頭所使用的機器人投顧做出拋售指令，或將促發一系列機器人投顧拋售，繼而導致市場崩盤。當機器人都在拋售，而沒有人在買的時候，崩盤將變得格外慘烈。

目前，AI 技术在银行业中的应用愈发广泛，主要集中在知识管理、身份认证、市场分析、客户关系管理、反洗钱和风控等方面。例如，在银行业的营销上，AI 技术可以对现有数据进行发掘进而识别出高收益客户的特征；在风险管理上，AI 技术能够对实时数据进行检测帮助银行实现反欺诈、反洗钱侦查；在知识管理上，AI 技术可以分析多维度的非结构化信息，对银行员工及客户提供相应的知识辅助等。

AML - [玩转日常消费领域的AI，为何在反恐道路上步履蹒跚？](https://www.jiqizhixin.com/articles/2017-08-20-2)

金融風控的痛點: 我一直認為，「科技進步是被業務需求逼出來的」。過去我們在互聯網行業靠算法和機器，都是被逼的，為什麼，因為數據量實在太大了，你想去淘寶搜個手機殼，讓阿里的同學人肉從上億的商品裡幫你找出最喜歡最合適的，那根本不可能。傳統金融場景裡，一筆100萬的貸款主要靠風控人員和關係，那是可行的；而到了銀行的信用卡中心，積壓的申請審核，讓審批人員每週加班，都批不完。那現在互聯網金融要面臨更加普惠的場景，比如幾百塊錢一筆的手機貸，靠鋪人力一定是行不通的。所以，這已經不單單是提升運營效率問題，而是必須要把活兒交給機器，讓機器來學習人的風控經驗，機器人變成風控專家。

金融領域應用機器學習與人工智能的難點第一個問題是數據太少。因為金融數據非常稀疏，而且現在的很多金融產品形式在以前沒有發生過，所沒有十幾年的數據積累。換句話說就是缺少訓練數據，這又被稱為冷啟動，缺數據。另外，金融領域出現壞賬情況少則一個月多則數月，數據積累需要等很久，相比之下，互聯網搜索領域內可以迅速拿到點擊反饋，兩者差別很大。所以數據缺失是阻礙機器來學習人類經驗的巨大障礙。

第二個是數據太多。這裡指的是數據特徵維度多，超過了人的處理能力。傳統金融只有十幾維度的特徵變量，人工調公式即可應對。但現在面臨這麼多維度的數據，大家也想了很多很好的願景，討論很多數據都可以用。但為什麼用不上呢？問題在於我們有什麼辦法可以有一個很強的表達能力將這些很原始的，也可以叫弱變量的數據特徵利用起來。將弱特徵數據組合起來，與結果聯繫起來，讓人的直觀經驗可以理解，讓風控專家去反饋。

在金融場景內，不能像互聯網機器學習一樣是一個黑盒子，一堆數據扔進去，等結果來反饋迭代。金融場景內，特別強調模型的可解釋性，這樣才能把人的風控經驗和直觀感受跟數據表現結果關聯起來。在此基礎上，我們才能說把人的經驗介入到利用數據進行機器學習建模的操作中去。做到特徵要能夠追溯回去，尤其是金融的反饋結果要等很久，需要人能夠快速乾預反饋。

如何解決金融風控冷啟動問題

數據太少

對於數據太少和產生太慢的問題，冷啟動問題是一個非常典型的case。我們在互聯網行業經常面臨缺少數據的問題，也積累了成熟的經驗，就是把人的因素疊加到機器學習過程中去。我們做搜索廣告時，會請人標註數據，然後通過標註數據的專家來指導算法工程師調優算法，改進排序結果。而在金融場景裡，我們有很多現成的經驗以及經驗豐富的風控人員，這些專家有很強的風控知識。

理論上講，如果有幾百個風控專家，不用發工資，我們做手機貸也可以做下去，但實際情況是我們必須靠機器去學習人的風控經驗。所以我們通過半監督學習的方法，把業務風控專家和實際的信貸結果在online學習中做一個結合。在這個過程中，風控人員可以實時的介入，不停地根據輸出結果做一些調整，然後非常實時地反饋到模型訓練的迭代提升的過程當中。

這就說我們特別重視人的因素。現在大家都在講人工智能，人工智能的本質是什麼？在我的理解其實就是讓機器學習人的經驗。以前我們依賴幾個經驗豐富的風控人員，現在我們可以讓機器把人的經驗學過來，然後讓機器來做一個自動的決策。

金融的業務結果和樣本非常珍貴。比如，我之前在房貸業務上積累了一些樣本，然後換到一個新的消費信貸業務上，或者從一個消費信貸業務切換到另一個新的業務。這些珍貴的樣本數據不能丟掉，但怎麼去用呢？我們可以做到儘可能利用已有的經驗和知識，把generic的風險核心模型和domain knowledge分開，再去根據業務的場景信息，以及場景內的先驗知識結合起來，在此基礎上學習和復用跨領域跨場景的知識，並且可以做到知識積累。

深度學習技術解決特徵工程的難點數據太多

接下來我們來看「數據太多」。我將這個問題分為兩個部分來看。

首先是數據的特徵維度很多。我們關心的是如何將大數據和金融風控的問題掛鉤起來，這裡面其實是需要非常強大的特徵加工和表達能力。這是傳統的線性回歸統計建模方法很難去完成的。我們的辦法有很多，這裡面包括大家現在熱度很高的「深度學習」。深度學習的本質是通過數據特徵的處理去學習人處理知識和數據的方式。為瞭解決數據太多的問題，讓人能看透浩瀚的原始數據，在模型的前端，我們嘗試了不同的深度特徵編碼方法，非監督學習的方法對原始數據進行預處理，從而實現特徵的降維，將浩瀚的原始數據和最後結果掛上鉤。

模型的可解釋性

其次是模型的可解釋性。金融專家特別關心模型的可解釋性。這裡面有兩個意義：

如果給信貸對象一個打分的結果，如果不能解釋，這個很難和申請人去溝通的；

另外，我們所面臨的是一個非常複雜的環境，如果對於風控結果仍然是黑盒進黑盒出的話，風險是很難去把控和估計的。

如果模型出了問題，造成的風險漏洞是我們不能承受的。在互聯網金融業務這麼快速成長的背景下，很有可能公司的業務都做不下去。所以，互聯網內黑盒進黑盒出的方法就不適用於金融場景，需要有一個可解釋的local模型去做到。我們的實踐經驗是，利用LIME去捕獲結果或者局部結果中的關鍵變量，然後讓風控專家迅速的抓到是哪些特徵導致結果的變化。

氪信取得的效果我們把互聯網的技術經驗，在金融場景內做了一些艱難的嘗試，並得到了一些實踐經驗，包括從最開始的數據獲取處理，到人的介入參與，到對複雜模型的干預過程，最後形成我們的practice。

從效率上說，我們的一個合作夥伴得到了很好的效果。他們做了一個金融信貸場景，部署在氪信的系統和模型上跑，只需要3-4個業務風控兼運營的人員，風控的大部分工作交給機器去做。

另外從效果上看，我們利用DNN模型做出來一個結果，可以看到結果比傳統的LR模型ks值從0.19提升到0.43。數字和結果是我們做模型的人最直接的一個答案，這裡面沒有什麼可以講概念的。

大家之前對大數據期望值很高，又屢屢失望，現在其實對數據科技來說是一個很好的時機。因為大家真的需要能夠有運用數據的能力，用機器解決金融實際問題，這也是我們這個時代的機會和風口，也是一個新的開始。

下個月的12、13號，雷鋒網將在深圳舉辦一場盛況空前的人工智能與機器人峰會，屆時我們將發佈「人工智能&機器人Top25創新企業榜」榜單，為此我們在蒐集並確認AI、機器人、自動駕駛、無人機等幾個領域的優質項目。

### Compliance

[Digital Reasoning](http://www.digitalreasoning.com/), a private analytics firm based in Nashville, machine learning techniques are developed to provide proactive compliance analytics, completing tasks such as sifting through employee emails for potentially non- compliant content and detecting violations such as market manipulation, unauthorized trading or wall cross violation.

### Credit Risk Reduction

1. forecast credit delinquencies
2. fraud detection

### Creative

* 过去一个养殖户用 12 块钱养一只鸡，需要提前贷款，贷一笔很大的款放在那儿，这个时候就产生了很大利息支出，这对养鸡户是一笔很大的费用。但是在鸡成长过程中，小鸡不需要吃那么多饲料，只有长大的时候才吃。不同的鸡对不同饲料需求也不一样。当我们能够掌握这个过程的时候，我们就可以把放贷变得因地制宜，在一开始养鸡户不需要借这么多钱，过去逐次增加。所以如果我们能够做到按日计息就可以做到这一点，就能够大大提升效率。现在京东金融可以做到 6 分钱就养一只鸡，用人工智能算法来做分析。
* 帮助养猪户鉴别不同的猪在养殖过程中的各种活动，这个猪是不是活跃，它是不是散养的或者它还是一只非常不爱动的猪，由此对健康就可以作决策，对保险就可以作预测。

## Insurance

According to [DataRobot](https://www.datarobot.com/use-cases/#insurance):

* Life Insurance Underwriting for Impaired Life Customers
* Insurance Pricing
* Fraudulent Claim Modeling
* Conversion Modeling
* Claim Payment Automation Modeling
* Claim Development Modeling

智能手機應用、客戶活動可穿戴設備、索賠優化工具、個人客戶風險發展系統、在線保單處理、制動化的合規程序

大多數的 InsurTech 都是在 2011 年開始發力，僅僅在這一年，45 位投資人總投資額就到達了 1.31 億美元。如今，InsurTech 是一個價值 26 億美元的子產業，正變革或助力行業既得利益獲得者。如同 FinTech 一樣，InsurTech 領域的全球投資巨幅增長，自從 2011 年以來增長了 1900%（ 2011 年的 1.31 億美元到 2016 年的 26 億美元\)。在未來幾年，InsurTech 將繼續保持增長勢頭。因為在保險上，消費者傾向於更加個性化，容易使用的保險產品。

採用新的技術對保險人來說並不是一件容易的事情，在未來幾年有很多的障礙需要克服。例如：複雜的保險流程，比如健康險的索賠，需要面對面或者電話交流；數據的數字化儲存可能會導致隱私問題或黑客攻擊等；海嘯，火災，地震，洪水作為不可預測的災害仍將繼續困擾保險人；政府的規則束縛需要企業、保險公司和立法機關的協同作用。在 RiskMinds 大會（全球風險管理大會）上，當主辦方問道，InsurTech 對保險業是摧毀還是拯救的時候，各個保險業內人士說法不一。但所有保險的 CRO, CTO 和首席精算師都一致認為，InsurTech 正在改變保險的面貌。

初創公司 Cape Analytics 是一個非常好的案例。這家公司憑藉 1400 萬美元資金，將機器學習與計算機視覺及地理空間圖像結合，開發了一種為財產保險公司進行服務的解決方案，幫助分析房屋屋頂材料類型、狀況和建築物的佔地面積等。

再保險數據分析QuanTemplate 成立於 2013 年，總部位於直布羅陀，獲得了保險巨頭安聯集團等投資的 1025 萬美元資金。QuanTemplate 是一家保險數據管理和分析公司，為保險公司提供解決方案，幫助保險公司從保費和索賠中瞭解其投資組合數據，確認利弊風險點，衡量和監督承保業績，起草符合管理或監管的報告如 Solvency II pillar 3（Solvency II（2009/138 / EC）是歐盟法律中的一項指令，用於對歐盟保險條例進行編纂和協調，規定了歐盟保險公司為降低破產風險而必須持有的資本。pillar 3 列出了保險公司風險管理的要求，以及對保險公司監管細則）。Quantemplate 解決方案基於保險公司的政策和索賠信息，集成第三方數據，為保險公司提供大量有用的見解。顯然，它在再保險領域也有特殊的應用。保險公司出於減輕自身承擔的保險責任的目的，會用「再保險」的方式將自己不願意承擔或超過自己承保能力的保險責任轉嫁給其他保險公司。

利用社交媒體的實時投保加州初創公司 Carpe Data 成立於 2016 年，一輪的融資額已經達到 660 萬美元。該公司為承保項目提供預測性評分方案，如個人、汽車、人壽，甚至無法通過傳統商業置信評估的小型或微型業務。評估分數覆蓋美容、水療、汽車修理、健身、餐飲和酒店等領域的 2800 萬個美國小型企業。系統還會對客戶生活中發生的重大事件發出實時報備，如婚姻，分娩，工作變更或房屋購買等，這就相當於為保險公司提供了重新參與及覆蓋顧客的絕佳機會。Carpe Data 能利用任何「社交網絡、在線內容、可穿戴設備、連接設備和其他形式的下一代數據」。因此，在未來的某個時刻，您的保費可能會由 Facebook 來定。

## Retail

* 洞察客户数据：机器学习可以挖掘销售公司所收集到的客户数据背后的信息。研究表明，对客户数据的理解是必不可少的。虽然许多公司有一套完整的系统，在收集和存储客户数据上也在投入了大量的资源和人力，但是机器学习可以帮助销售公司更有效率地利用数据。
* 提升销售预测能力：销售公司可以借助机器学习算法，与历史销售情况进行对比，从而对最优解决方案、交易完成的可能性以及需要消耗的时间做出更好的预测。这个过程能够帮助销售经理更好地进行资源分配、完成销售方案规划。
* 预测消费者的需求：一家企业能否成功往往取决于是否能够更好地满足客户需求。机器学习可以帮助企业提升对客户需求的敏感性和主动性。在客户获得更好的购买建议之前，如果企业能够了解客户在销售方面的需求，企业与客户之间的关系就会愈发紧密。与人不同的是，机器在这个过程中从不会忘记后续跟进，也不会因为分身乏术而无法主动与客户分享解决方案。
* 有效的交易型销售：根据哈佛商业评论，到 2020 年，85% 的消费者互动将不再需要人工来进行。机器的引入可以快速、有效地处理当前的销售状况，同时能够解放销售人员，让他们可以更加关注于与客户的关系。
* 销售沟通：机器学习很有可能会引起销售沟通的巨大变革，比如借助短信等方式向客户提供智能回复。

## For good

刚刚过去的 7 月 30 日是世界打击贩运人口日。据统计，贩运人口已经成为全球性问题，每年至少有 250 万人成为人口贩运活动的受害者，沦为性工作者、乞丐、劳工等，非法利润约达 1500 亿美元。不过，人口贩运过程的本质上是一个供应链，借助技术手段对供应链中留下的数据线索进行分析，洞察数据背后的信息，可以帮助执法人员打击人口贩运行为。首先，可以通过大数据确定人群中人口贩运风险指数较高的个体，并针对这些目标人物执行盯防计划。风险指数可能涉及贫穷、失业、移民等方面的考量，同时，有组织犯罪和自然灾害的经历也可能会改变一个人的风险指数。另外，研究人员可以追踪不同地点的数据并加以分析，对受害者进行识别和定位。例如，贩卖者可能在社交媒体和互联网站点上做广告，执法人员可以利用面部识别软件将对失踪人口与广告中的照片加以比对。在这个过程中，机器学习技术有助于清除伪造的信息、预测丢失的数据。除此之外，网络分析可以用来描绘 Facebook、Twitter 等社交网络中的用户形象及相关动态，有助于识别出高危人群，如贩运者或买家，打击贩运交易链。

## Other industries

* autonomous driving
* image annotation
* game-playing
* [helicopter flying](http://people.eecs.berkeley.edu/~jordan/papers/ng-etal03.pdf)
* [speech synthesis](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
* movie recommendation
* Health: [improving diagnosis](https://www.humandx.org/), [predicting outbreaks of dengue](http://aime.life/), reducing inappropriate antibiotics use\]\([http://www.accurx.com/home](http://www.accurx.com/home)\)
* Global poverty: [identifying appropriate customers for pay-as-you-go solar electricity](http://web.archive.org/web/20170615121008/http://www.datakind.org/projects/machine-learning-to-help-rural-households-access-electricity), [using satellite imagery to find the right villages to receive cash transfers](http://web.archive.org/web/20160714072759/http://www.datakind.org/projects/using-the-simple-to-be-radical/)
* Animal welfare: [predicting which plant proteins will be best for making plant-based meat](http://web.archive.org/web/20170614185031/http://www.economist.com/news/technology-quarterly/21645497-tech-startups-are-moving-food-business-make-sustainable-versions-meat)

## For all industries

* respond to employee technology service desk requests through a natural language interface: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf).
* 精准营销
* 智能客服
  * optimizing clientservicing channels: Per page 49 of [JPMorgan's 2016 Annual Shareholder Report](https://www.jpmorganchase.com/corporate/investor-relations/document/2016-annualreport.pdf).
* cybersecurity
* Direct Marketing: Use DataRobot to predict which people have the highest likelihood of purchasing your product or service, providing the most cost-effective target marketing.
  * Challenge: Marketing to prospects \(direct mail, telemarketing and email\) is expensive. Targeting incorrectly can hurt your brand, leaving prospects feeling spammed. Traditional techniques are not very sophisticated, resulting in low response rates which in turn leads to high cost-per-lead/acquisition numbers.
  * Solution: Modern machine learning algorithms can substantially increase accuracy for determining which prospects should and should not receive your marketing material, yielding higher ROI. To drive down your cost-per-lead/acquisition numbers, use DataRobot's automated machine learning techniques to bring more sophistication to your marketing. By communicating with only those with a high likelihood to respond \(those who actually want to receive the material\) you are maintaining a favorable image of your company and saving money.

## Startups

### Orbital Insight: Pioneering use of AI in satellite imagery analytics

We spoke with Orbital Insight, a Palo Alto, CA based data analytics company that aggregates satellite imagery data from eight satellite providers and uses artificial intelligence technology to accelerate go-to-market applicability for asset managers.The Problem: Asset management firms are facing an increasingly competitive landscape as technological advance universalizes access to data and accelerates market reactions to one-off events. As firms look for ways to exploit market inefficiencies, many vast, relevant data sources remain untapped \(i.e. satellite imagery, shipping movement\) or inefficiently commercialized for market use.Orbital Insight solution: Orbital uses satellite data to isolate images that uniquely inform specific market trends. Whether aggregating the shadow shapes on the lids of oil drums to inform commodities prices or quantifying retail traffic patterns for major retailers, the company’s analytics solution leverages vast data sets, often in areas out of the reach of traditional collection metrics, and trains machine learning algorithms to quickly package data relevant to desired solutions. While the company noted that image data itself is publically available for purchase, its ability to leverage proprietary machine learning which advances beyond merely academic use-cases is paramount to creating differentiated insights into the implications that data has for investors.The company expressed the difficulty of leveraging today’s satellite images, as satellite revisit rates to any specific location range from 15-30 days. This requires normalization for variability in an image’s relative time of capture as well as other relevant shifts in control variables. A recent partnership with Planet Labs, however, could allow the company to gain access to data sets that provide daily images for every part of the world by next year, as fleets of nanosatellites are set to enter orbit.Orbital indicated that its value proposition is highlighted through its 50-60 proprietary neural network classifiers, which are essentially algorithms trained on “training sets” to seek out and identify points of interest and specific characteristics about the points of interest. The company estimates that their deep learning algorithms have reached 90- 95% accuracy today, validating AI predictions by using reliable data sets \(i.e. EIA oil storage data\) for comparison.The intersection of the cloud and AI reduces bottlenecks: As Orbital scales and gains access to more and more image data, it is utilizing Amazon Web Services’ \(AWS\) cloud platform to temporarily store data while it is being analyzed. Given the potential storage clog of images, especially given the partnership with Planet Labs, the company indicated that it would rely on the quick, efficient turnover of data by the AI system and balance a consistent inflow/ outflow of image inventory as projects are completed.

\(from Goldman - AI, Machine Learning and Data Fuel the Future of Productivity\)

