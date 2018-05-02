# New York City Taxi Trip Duration

 

> It's important to note though that average speed is a function of distance and time so it wouldn't add anything to the modelling output. We'll therefore need to remove it eventually before we train our model.

> Something definitely worth exploring, which could boost the performance of the XGBoost model significantly, is to create a data set that can be used with \[Xiaolin Wu's line algorithm\]\(https://en.m.wikipedia.org/wiki/Xiaolin\_Wu%27s\_line\_algorithm "Xiaolin Wu's line algorithm"\). This would involve pixelating the graph area and recording every pixel crossed by the line from the pick-up location to the drop-off location. If you can make the resolution as high as possible some of the pixels shoudl encapsulate traffic junctions, traffic lights, bridges, etc. Using the "has crossed coordinate X" features one could potentially create an extra +-10 000 features to train the alogrithm with.

\([KarelVerhoeven](https://www.kaggle.com/karelrv)[NYCT](https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial)\)

