# transformer-time-series

In this research, two "Transformer" architectures will be used, namely the "Transformer" Encoder-Decoder architecture and the "Transformer" FC layer which only uses the encoder part. In this study, several different hyperparameters will be tested, using different numbers of heads for about 8 and 16 and experiments with different layer hyperparameters. 

## Result 

## Training data with Transformer Encoder-Decoder and Transformer Encoder- FC layer
<img width="539" alt="image" src="https://github.com/ikacandraw/transformer-time-series/assets/99155423/d4a25098-b525-441e-b809-49ed55b5ad9f">

The RMSE and MAPE results for each model combination for each architecture show that there is not much difference, so it can be concluded that the use of different combinations of heads and layers in the Transformer architecture does not have a very significant effect on the prediction results or forecast results. A differentiator that can provide significant results is the use of different architectures for modeling time-series data. So for forecasting, the model that can be chosen is the "Transformer" encoder-FC layer model with 8 heads and 5 layers. To find out more, the RMSE and MAPE values for the "Transformer" model.

## Transformer with Encode FC-layer

<img width="305" alt="image" src="https://github.com/ikacandraw/transformer-time-series/assets/99155423/cf548ae5-45b9-4c69-a0e5-5fac29cf87c4">

the model that has the smallest RMSE and MAPE is the model with a combination of 8 heads and 6 layers.
