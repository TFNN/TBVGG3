# TBVGG3

---
### Oh bias, what did I do?
Looking back at this code a year after publishing it online I noticed my implementation of bias per filter was a little odd and I can remember my reasoning to some extent for implementing it that way, the problem is I don't like to second guess my previous self; I know that a year ago I thought a lot about how to implement the bias, more than I have thought now about if it was the right way or not. So I've left the implementation the way it was originally on all versions of this code apart from in [TBVGG3_ADA_MED.h](TBVGG3_ADA_MED.h) where I have implemented a new method of bias that I deem to probably be more correct. Also, I think the `TBVGG3_ADA_MED` version is a better scaling of filters per layer and these days I just prefer ADAGRAD as a defacto optimiser.

---

A Convolutional Neural Network Library written in C

Please refer to [TFCNN/Projects](https://github.com/TFCNN/Projects) for an example use case.

To learn more about the VGG Network coined by the [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) the Tensorflow Keras documentation has good coverage of the two mainstream options VGG16 and VGG19 [here](https://keras.io/api/applications/vgg/).

This is not associated with the Oxford Visual Geometry Group, the network is inspired by the VGG network, it is essentially a very similar type of network because it uses only 3x3 kernels. The number following VGG is the number of layers in the network, hence this network having been coined VGG3, and TBVGG3 because it has a Tiny footprint and a Binary output; making it the Tiny Binary VGG 3 (TBVGG3).

### TBVGG3 has a very simple interface
_TBVGG3 is designed to take a 28x28 image with 3 colour channels (RGB) as input._
```
enum 
{
    LEARN_MAX = 1,
    LEARN_MIN = 0,
    NO_LEARN  = -1
}
typedef TBVGG3_LEARNTYPE;

float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn);
void  TBVGG3_Reset(TBVGG3_Network* net);
int   TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file);
int   TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file);
```

### Example Usage
```
TBVGG3_Network net;
TBVGG3_Reset(&net) // this will initialise the weights
// now you can train the network with TBVGG3_Process(&net, &inputs, LEARN_MAX)
// or query a trained network with TBVGG3_Process(&net, &inputs, NO_LEARN)
```
_It is important that you always call `TBVGG3_Reset()` on a newly created network to initialise the weights._ 
