These all use the original bias implementation with a sigmoid output.

I have depreciated the use of a sigmoid output for a linear layer and it seems to hinder the training process.

The original bias implementation used here is probably wrong, but it work fine back in January 2021 with the original release of [TBVGG3_SGD.h](TBVGG3_SGD.h) and [TBVGG3_NAG.h](TBVGG3_NAG.h) so I'm keeping it around as a spectacle of guesswork.
