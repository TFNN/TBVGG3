# TBVGG3
A Convolutional Neural Network Library written in C

Please refer to [TFCNN/Projects](https://github.com/TFCNN/Projects) for an example usecase.

To learn more about the VGG Network coined by the [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) the Tensorflow Keras documentation has a good coverage of the two mainstream options VGG16 and VGG19 [here](https://keras.io/api/applications/vgg/).

This is no associated to the Oxford Visual Geometry Group, but the network was inspired by the VGG network, it is essentially a very similar type of network because it uses only 3x3 kernels. The number following VGG is the number of layers in the network, hence this network having been coined VGG3, and TBVGG3 because it has a Tiny footprint and a Binary output; making it the Tiny Binary VGG 3.

TBVGG3 has a very simple interface:
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

