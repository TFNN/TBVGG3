---
### Oh bias, what did I do?
Looking back at this code over a year after publishing it online I noticed my implementation of bias per filter was a little odd and I can remember my reasoning to some extent for implementing it that way, the problem is I don't like to second guess my previous self; I know that a year ago I thought a lot about how to implement the bias, more than I have thought now about if it was the right way or not. So I've left the implementation the way it was originally on all versions of this code apart from in [TBVGG3_ADA_MED.h](TBVGG3_ADA_MED.h) where I have implemented a new method of bias that I deem to probably be more correct. Also, I think the `TBVGG3_ADA_MED` version is a better scaling of filters per layer and these days I just prefer [ADAGRAD](https://machinelearningmastery.com/gradient-descent-with-adagrad-from-scratch/) as a defacto optimiser.
---

These all use the original bias implementation with a sigmoid output.

I have depreciated the use of a sigmoid output for a linear layer and it seems to hinder the training process.

The original bias implementation used here is probably wrong, but it work fine back in January 2021 with the original release of [TBVGG3_SGD.h](TBVGG3_SGD.h) and [TBVGG3_NAG.h](TBVGG3_NAG.h) so I'm keeping it around as a spectacle of guesswork.
