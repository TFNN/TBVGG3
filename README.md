# TBVGG3

A Fully Convolutional Network (FCN) written in C.

The dataset used in the example provided [main.c](main.c) is [CSGO.zip](https://github.com/TFNN/DOCS/raw/main/DATASETS/CSGO.zip) from [DOCS/DATASETS](https://github.com/TFNN/DOCS/tree/main/DATASETS).

To learn more about the VGG Network coined by the [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) the Tensorflow Keras documentation has good coverage of the two mainstream options VGG16 and VGG19 [here](https://keras.io/api/applications/vgg/).

This is not associated with the Oxford Visual Geometry Group, the network is inspired by the VGG network, it is essentially a very similar type of network because it uses only 3x3 kernels. The number following VGG is the number of layers in the network, hence this network having been coined VGG3, and TBVGG3 because it has a Tiny footprint and a Binary output; making it the Tiny Binary VGG 3 (TBVGG3).

The network has three default sizes, `ADA8`, `ADA16`, and `ADA32` although you could setup bigger networks, it's unlikely you would need to. The reason for the ADA naming convention is because originaly I only intended to support the ADAGRAD optimiser, although since then I have added SGD & NAG and the naming convention ADA for the sizes of kernels/filters per layer multiples just stuck. In an ADA8 network for example, the first layer has 8 kernels, the second has 16 kernels and the third and last layer has 32 kernels. Each subsequent layer has a squared amount of kernels from the last layer.

### TBVGG3 has a very simple interface
_TBVGG3 is designed to take a 28x28 image with 3 colour channels (RGB) as input, preferably normalised -1 to 1._
```
// options that can be defined before including TBVGG3.h
#define ADA8                // or default: ADA16
#define ADA32               // or default: ADA16
#define UNIFORM_GLOROT      // or default: NORMAL_GLOROT
#define OPTIM_SGD           // or default: OPTIM_ADA
#define OPTIM_NAG           // or default: OPTIM_ADA
#define SIGMOID_OUTPUT      // or default: linear output
#define LEARNING_RATE 0.01f // or default: 0.001f
#define GAIN 1.f            // or default: 0.0065f
#define NAG_MOMENTUM 0.9f   // or default: 0.1f

#define TBVGG3_LEARNTYPE float
#define LEARN_MAX 1.f
#define LEARN_MIN 0.f
#define NO_LEARN -1.f // pass this when you only require a forward pass.
float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn);

void  TBVGG3_Reset(TBVGG3_Network* net, uint seed);
int   TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file);
int   TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file);

void  TBVGG3_Debug(TBVGG3_Network* net); // print layer stats
```

### Example Usage
```
TBVGG3_Network net;
TBVGG3_Reset(&net); // this will initialise the weights
// now you can train the network with TBVGG3_Process(&net, &inputs, LEARN_MAX);
// or query a trained network with TBVGG3_Process(&net, &inputs, NO_LEARN);
```
_It is important that you always call `TBVGG3_Reset();` on a newly created network to initialise the weights._ 

_An implementation is available [here](https://github.com/jcwml/CSGO-Trigger-Bot) and [here](https://github.com/jcwml/CSGO-Trigger-Bot-2)._
