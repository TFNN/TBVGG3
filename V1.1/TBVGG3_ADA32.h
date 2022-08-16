/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
        AUGUST 2022 - TBVGG3_ADA64 v1.1
--------------------------------------------------
    Tiny Binary VGG3
    https://github.com/tfcnn
    
    This version uses the ADAGRAD optimisation algorithm.

    v1.1: Saved network no longer includes output and error layers.
    
    Release Notes:
        1. I have revised the bias implementation.

        2. Output is now a linear layer with sigmoid now optional
        by specifying `#define SIGMOID_OUTPUT`.

        3. You can now select between NORMAL_GLOROT or UNIFORM_GLOROT
        weight initialisation by specifying `#define UNIFORM_GLOROT`
        for uniform, otherwise normal is used by default.

        Sigmoid output is better for normalised inputs and a linear
        output is better for unnormalised inputs.

    This is an adaption inspired by the VGG series of networks.

    This VGG network is designed for binary classification and is
    only three layers deep. It uses Global Average Pooling rather
    than a final fully connected layer, additionally the final
    result is again just an average of the GAP. Essentially making
    this network an FCN version of the VGG network.

    The VGG network was originally created by the Visual Geometry Group
    of Oxford University in the United Kingdom. It was first proposed
    by Karen Simonyan and Andrew Zisserman, the original paper is
    available here; https://arxiv.org/abs/1409.1556

        TBVGG3
        :: ReLU + 0 Padding
        28x28 x32
        > maxpool
        14x14 x64
        > maxpool
        7x7 x128
        > GAP + Average

    I like to call the gradient the error, I am one of those.

    Configuration;
        No batching of the forward passes before backproping.

        XAVIER GLOROT normal distribution weight initialisation.
        I read some places online that uniform GLOROT works
        better in CNN's, this is something I really need to
        benchmark for myself at some point. Since the original
        VGG paper references GLOROT with normal distribution,
        this is what I chose initially.

        expected input RGB 28x28 pixels;
        float input[3][28][28];

    Preferences;
        You can see that I do not make an active effort to avoid
        branching, when I consider the trade off, such as with the
        TBVGG3_CheckPadded() check, I think to myself do I memcpy()
        to a new buffer with padding or include the padding in the
        original buffer or use branches to check if entering a padded
        coordinate, I chose the latter. I would rather a few extra
        branches than to bloat memory in some scenarios, although
        you can also see in TBVGG3_2x2MaxPool() that I choose a
        negligibly higher use of memory to avoid ALU divisions.

        I didn't think it was a good idea to maxpool the last
        layer because there are no fully connected layers,
        since it's going straight into a GAP it will make
        negligible difference in the final average. Maxpooling
        before a fully connected layer makes sense to reduce the
        amount of parameters to a more important subset. But this
        is a binary decision network, so a fully connected layer
        wont have a profound impact, we just want to know if our
        relevant features / filters had been activated enough to
        signal YES, if not, it's a NO.

    Comments;
        Do I think Bias makes a significant difference? It certainly
        seems to make the network train faster, it is extra parameters
        and 'hassle' to add to a network, it did make me think a little
        as to how best it would be implemented.

        When it came to the back propagation I just worked it out
        using the knowledge and intuition I had gained from implementing
        back propagation in Fully Connected Neural Networks which is a
        in my opinion easier to understand. That's to say I didn't read
        or check any existing documentation for implementing back prop
        in CNN's. To be honest, the problem is something you can just
        see in your minds eye when you think about it. You know that
        you have to push a gradient backward and that process is very
        much the same as in Fully Connected layers.

        It certainly feels like this CNN trains better on image data
        than an FNN although I do not feel that it is particularly
        easy to train in the real-time manner that I originally proposed
        here, I am going to put some more thought into improving the
        real-time training process but for this existing system
        the only option to improve the training is to variate to order
        of the objects you train on in real-time as if you where feeding
        a neural network for offline training, and also to fiddle the
        `LEARNING_RATE` and `GAIN` hyperparameters.
        
        When a ReLU output is fed into a regular Sigmoid function the
        output of the ReLU will always be >0 and thus the output of the
        Sigmoid will always be 0.5 - 1.0, and the derivative will start
        at 0.25 and then reduce to 0 as the sigmoid input approaches 1.
        As such I have provided a suggested modification to the Sigmoid
        function `1-(1 / expf(x))` which will insure that the output ranges
        from 0 to 1 and that the derivative will output 0.25 with an input
        of 0.5.

    Network size:
        364.2 KiB (372,992 bytes)
*/

#ifndef TBVGG3_H
#define TBVGG3_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef LINUX_DEBUG
    #include <sys/stat.h>
#endif

#define uint unsigned int
#define sint int
#define LEARNING_RATE 0.001f
#define GAIN          0.0065f

/*
--------------------------------------
    structures
--------------------------------------
*/

// network struct
struct
{
    //filters:num, d,  w
    float l1f[32][3 ][9];
    float l2f[64][32][9];
    float l3f[128][64][9];

    // filter bias's
    float l1fb[32][1];
    float l2fb[64][1];
    float l3fb[128][1];
}
typedef TBVGG3_Network;

#define TBVGG3_LEARNTYPE float
#define LEARN_MAX 1.f
#define LEARN_MIN 0.f
#define NO_LEARN -1.f

/*
--------------------------------------
    functions
--------------------------------------
*/

float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn);
void  TBVGG3_Reset(TBVGG3_Network* net);
int   TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file);
int   TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file);

#ifdef LINUX_DEBUG
    void TBVGG3_Dump(TBVGG3_Network* net, const char* folder);
#endif

/*
--------------------------------------
    the code ...
--------------------------------------
*/

#ifdef LINUX_DEBUG
void TBVGG3_Dump(TBVGG3_Network* net, const char* folder)
{
    char p[256];
    mkdir(folder, 0777);
    sprintf(p, "%s/l1f.txt", folder);
    FILE* f = fopen(p, "w");
    if(f != NULL){
        for(uint i = 0; i < 32; i++){
            fprintf(f, "~~~~~~~~~~~~~~N(%u):\n", i);
            for(uint j = 0; j < 3; j++){
                fprintf(f, "D(%u): ", j);
                for(uint k = 0; k < 9; k++)
                    fprintf(f, "%.2f ", net->l1f[i][j][k]);
                fprintf(f, ":: %f\n", net->l1fb[i][0]);
            }fprintf(f, "\n");
        }fclose(f);}
    sprintf(p, "%s/l2f.txt", folder);
    f = fopen(p, "w");
    if(f != NULL){
        for(uint i = 0; i < 64; i++){
            fprintf(f, "~~~~~~~~~~~~~~N(%u):\n", i);
            for(uint j = 0; j < 32; j++){
                fprintf(f, "D(%u): ", j);
                for(uint k = 0; k < 9; k++)
                    fprintf(f, "%.2f ", net->l2f[i][j][k]);
                fprintf(f, ":: %f\n", net->l2fb[i][0]);
            }fprintf(f, "\n");
        }fclose(f);}
    sprintf(p, "%s/l3f.txt", folder);
    f = fopen(p, "w");
    if(f != NULL){
        for(uint i = 0; i < 128; i++){
            fprintf(f, "~~~~~~~~~~~~~~N(%u):\n", i);
            for(uint j = 0; j < 64; j++){
                fprintf(f, "D(%u): ", j);
                for(uint k = 0; k < 9; k++)
                    fprintf(f, "%.2f ", net->l3f[i][j][k]);
                fprintf(f, ":: %f\n", net->l3fb[i][0]);
            }fprintf(f, "\n");
        }fclose(f);}
}
#endif

static inline float TBVGG3_RELU(const float x)
{
    if(x < 0.f){return 0.f;}
    return x;
}

static inline float TBVGG3_RELU_D(const float x)
{
    if(x > 0.f){return 1.f;}
    return 0.f;
}

#ifdef SIGMOID_OUTPUT
static inline float TBVGG3_SIGMOID(const float x)
{
    return 1.f-(1.f / expf(x));
    //return 1.f / (1.f + expf(-x));
}

static inline float TBVGG3_SIGMOID_D(const float x)
{
    return x * (1.f - x);
}
#endif

static inline float TBVGG3_ADA(const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] += err * err;
    return (LEARNING_RATE / sqrtf(momentum[0] + 1e-7f)) * err;
}

#ifdef UNIFORM_GLOROT
float TBVGG3_UniformRandom()
{
    static const float rmax = (float)RAND_MAX;
    float pr = 0.f;
    while(pr == 0.f) //never return 0
    {
        const float rv2 = ( ( (((float)rand())+1e-7f) / rmax ) * 2.f ) - 1.f;
        pr = roundf(rv2 * 100.f) / 100.f; // two decimals of precision
    }
    return pr;
}
#else
float TBVGG3_NormalRandom() // Box Muller
{
    static const float rmax = (float)RAND_MAX;
    float u = ( (((float)rand())+1e-7f) / rmax) * 2.f - 1.f;
    float v = ( (((float)rand())+1e-7f) / rmax) * 2.f - 1.f;
    float r = u * u + v * v;
    while(r == 0.f || r > 1.f)
    {
        u = ( (((float)rand())+1e-7f) / rmax) * 2.f - 1.f;
        v = ( (((float)rand())+1e-7f) / rmax) * 2.f - 1.f;
        r = u * u + v * v;
    }
    return u * sqrtf(-2.f * logf(r) / r);
}
#endif

void TBVGG3_Reset(TBVGG3_Network* net)
{
    if(net == NULL){return;}

    // seed random
    srand(time(0));

    // XAVIER GLOROT NORMAL
    // Weight Init

#ifdef UNIFORM_GLOROT
    //l1f
    float d = sqrtf(6.0f / 35.f);
    for(uint i = 0; i < 32; i++)
        for(uint j = 0; j < 3; j++)
            for(uint k = 0; k < 9; k++)
                net->l1f[i][j][k] = TBVGG3_UniformRandom() * d;

    //l2f
    d = sqrtf(6.0f / 96.f);
    for(uint i = 0; i < 64; i++)
        for(uint j = 0; j < 32; j++)
            for(uint k = 0; k < 9; k++)
                net->l2f[i][j][k] = TBVGG3_UniformRandom() * d;

    //l3f
    d = sqrtf(6.0f / 192.f);
    for(uint i = 0; i < 128; i++)
        for(uint j = 0; j < 64; j++)
            for(uint k = 0; k < 9; k++)
                net->l3f[i][j][k] = TBVGG3_UniformRandom() * d;
#else
    //l1f
    float d = sqrtf(2.0f / 35.f);
    for(uint i = 0; i < 32; i++)
        for(uint j = 0; j < 3; j++)
            for(uint k = 0; k < 9; k++)
                net->l1f[i][j][k] = TBVGG3_NormalRandom() * d;

    //l2f
    d = sqrtf(2.0f / 96.f);
    for(uint i = 0; i < 64; i++)
        for(uint j = 0; j < 32; j++)
            for(uint k = 0; k < 9; k++)
                net->l2f[i][j][k] = TBVGG3_NormalRandom() * d;

    //l3f
    d = sqrtf(2.0f / 192.f);
    for(uint i = 0; i < 128; i++)
        for(uint j = 0; j < 64; j++)
            for(uint k = 0; k < 9; k++)
                net->l3f[i][j][k] = TBVGG3_NormalRandom() * d;
#endif

    // reset bias
    memset(net->l1fb, 0, sizeof(net->l1fb));
    memset(net->l2fb, 0, sizeof(net->l2fb));
    memset(net->l3fb, 0, sizeof(net->l3fb));
}

int TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "wb");
    if(f == NULL)
        return -1;

    if(fwrite(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

int TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    if(fread(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void TBVGG3_2x2MaxPool(const uint depth, const uint wh, const float input[depth][wh][wh], float output[depth][wh/2][wh/2])
{
    // for every depth
    for(uint d = 0; d < depth; d++)
    {
        // output tracking, more memory for less alu division ops
        uint oi = 0, oj = 0;

        // for every 2x2 chunk of input
        const uint wh1 = wh-1;
        for(uint i = 0; i < wh1; i += 2, oi++)
        {
            for(uint j = 0; j < wh1; j += 2, oj++)
            {
                // get max val
                float max = 0.f;
                if(input[d][i][j] > max)
                    max = input[d][i][j];
                if(input[d][i][j+1] > max)
                    max = input[d][i][j+1];
                if(input[d][i+1][j] > max)
                    max = input[d][i+1][j];
                if(input[d][i+1][j+1] > max)
                    max = input[d][i+1][j+1];

                // output max val
                output[d][oi][oj] = max;
            }
            oj = 0;
        }
    }
}

static inline uint TBVGG3_CheckPadded(const sint x, const sint y, const uint wh)
{
    if(x < 0 || y < 0 || x >= wh || y >= wh)
        return 1;
    return 0;
}

float TBVGG3_3x3Conv(const uint depth, const uint wh, const float input[depth][wh][wh], const uint y, const uint x, const float filter[depth][9], const float* filter_bias)
{
    // input depth needs to be same as filter depth
    // This will return a single float output. Call this x*y times per filter.
    // It's zero padded so if TBVGG3_CheckPadded() returns 1 it's a no operation
    float ro = 0.f;
    sint nx = 0, ny = 0;
    for(uint i = 0; i < depth; i++)
    {
        // lower row
        nx = x-1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][0];

        nx = x,   ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][1];

        nx = x+1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][2];

        // middle row
        nx = x-1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][3];

        nx = x,   ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][4];

        nx = x+1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][5];

        // top row
        nx = x-1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][6];

        nx = x,   ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][7];

        nx = x+1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][8];
    }

    // bias
    ro += filter_bias[0];

    // return output
    return TBVGG3_RELU(ro);
}

void TBVGG3_3x3ConvB(const uint depth, const uint wh, const float input[depth][wh][wh], const float error[depth][wh][wh], const uint y, const uint x, float filter[depth][9], float filter_momentum[depth][9], float* bias, float* bias_momentum)
{
    // backprop version
    sint nx = 0, ny = 0;
    for(uint i = 0; i < depth; i++)
    {
        // lower row
        nx = x-1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][0] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][0]);
            
        nx = x,   ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][1] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][1]);

        nx = x+1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][2] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][2]);

        // middle row
        nx = x-1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][3] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][3]);

        nx = x,   ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][4] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][4]);

        nx = x+1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][5] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][5]);

        // top row
        nx = x-1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][6] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][6]);

        nx = x,   ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][7] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][7]);

        nx = x+1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][8] += TBVGG3_ADA(input[i][ny][nx], error[i][y][x], &filter_momentum[i][8]);
        
        // bias
        bias[0] += TBVGG3_ADA(1, error[i][y][x], bias_momentum);
    }
}

float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn)
{
    if(net == NULL){return -1.f;}

    // filter momentum's
    float l1fm[32 ][3 ][9]={0};
    float l2fm[64 ][32][9]={0};
    float l3fm[128][64][9]={0};

    // filter bias momentum's
    float l1fbm[32 ][1]={0};
    float l2fbm[64 ][1]={0};
    float l3fbm[128][1]={0};

    // outputs
    //       d,  y,  x
    float o1[32][28][28];
        float p1[32][14][14]; // pooled
    float o2[64][14][14];
        float p2[64][7][7];   // pooled
    float o3[128][7][7];

    // error gradients
    //       d,  y,  x
    float e1[32][28][28];
    float e2[64][14][14];
    float e3[128][7][7];

    // convolve input with 32 filters
    for(uint i = 0; i < 32; i++) // num filter
    {
        for(uint j = 0; j < 28; j++) // height
        {
            for(uint k = 0; k < 28; k++) // width
            {
                o1[i][j][k] = TBVGG3_3x3Conv(3, 28, input, j, k, net->l1f[i], net->l1fb[i]);
            }
        }
    }

    // max pool the output
    TBVGG3_2x2MaxPool(32, 28, o1, p1);

    // convolve output with 64 filters
    for(uint i = 0; i < 64; i++) // num filter
    {
        for(uint j = 0; j < 14; j++) // height
        {
            for(uint k = 0; k < 14; k++) // width
            {
                o2[i][j][k] = TBVGG3_3x3Conv(32, 14, p1, j, k, net->l2f[i], net->l2fb[i]);
            }
        }
    }

    // max pool the output
    TBVGG3_2x2MaxPool(64, 14, o2, p2);

    // convolve output with 128 filters
    for(uint i = 0; i < 128; i++) // num filter
    {
        for(uint j = 0; j < 7; j++) // height
        {
            for(uint k = 0; k < 7; k++) // width
            {
                o3[i][j][k] = TBVGG3_3x3Conv(64, 7, p2, j, k, net->l3f[i], net->l3fb[i]);
            }
        }
    }

    // global average pooling
    float gap[128] = {0};
    for(uint i = 0; i < 128; i++)
    {
        for(uint j = 0; j < 7; j++)
            for(uint k = 0; k < 7; k++)
                gap[i] += o3[i][j][k];
        gap[i] /= 49.f;
    }

    // average final activation
    float output = 0.f;
    for(uint i = 0; i < 128; i++)
        output += gap[i];
    output /= 128.f;

#ifdef SIGMOID_OUTPUT
    output = TBVGG3_SIGMOID(output);
#endif

    // return activation else backprop
    if(learn == NO_LEARN)
    {
        return output;
    }
    else
    {
        // error/gradient slope scaled by derivative
#ifdef SIGMOID_OUTPUT
        const float g0 = TBVGG3_SIGMOID_D(output) * (learn - output);
        //printf("g0: %f %f %f %f %f\n", g0, learn, output, (learn - output), TBVGG3_SIGMOID_D(output));
#else
        float g0 = learn - output;
        //printf("g0: %f %f %f %f\n", g0, learn, output, (learn - output));
#endif

        // ********** Gradient Back Pass **********

        // layer 3
        float l3er = 0.f;
        for(uint i = 0; i < 128; i++) // num filter
        {
            for(uint j = 0; j < 7; j++) // height
            {
                for(uint k = 0; k < 7; k++) // width
                {
                    // set error
                    e3[i][j][k] = GAIN * TBVGG3_RELU_D(o3[i][j][k]) * g0;

                    // every output error gradient for every filter weight :: per filter
                    for(uint d = 0; d < 64; d++) // depth
                        for(uint w = 0; w < 9; w++) // weight
                            l3er += net->l3f[i][d][w] * e3[i][j][k];
                    l3er += net->l3fb[i][0] * e3[i][j][k];
                }
            }
        }

        // layer 2
        float l2er = 0.f;
        for(uint i = 0; i < 64; i++) // num filter
        {
            for(uint j = 0; j < 14; j++) // height
            {
                for(uint k = 0; k < 14; k++) // width
                {
                    // set error
                    e2[i][j][k] = GAIN * TBVGG3_RELU_D(o2[i][j][k]) * l3er;

                    // every output error gradient for every filter weight :: per filter
                    for(uint d = 0; d < 32; d++) // depth
                        for(uint w = 0; w < 9; w++) // weight
                            l2er += net->l2f[i][d][w] * e2[i][j][k];
                    l2er += net->l2fb[i][0] * e2[i][j][k];
                }
            }
        }

        // layer 1
        for(uint i = 0; i < 32; i++) // num filter
        {
            for(uint j = 0; j < 28; j++) // height
                for(uint k = 0; k < 28; k++) // width
                    e1[i][j][k] = GAIN * TBVGG3_RELU_D(o1[i][j][k]) * l2er; // set error
        }

        // ********** Weight Nudge Forward Pass **********
        
        // convolve filter 1 with layer 1 error gradients
        for(uint i = 0; i < 32; i++) // num filter
        {
            for(uint j = 0; j < 28; j++) // height
                for(uint k = 0; k < 28; k++) // width
                    TBVGG3_3x3ConvB(3, 28, input, e1, j, k, net->l1f[i], l1fm[i], net->l1fb[i], l1fbm[i]);
        }

        // convolve filter 2 with layer 2 error gradients
        for(uint i = 0; i < 64; i++) // num filter
        {
            for(uint j = 0; j < 14; j++) // height
                for(uint k = 0; k < 14; k++) // width
                    TBVGG3_3x3ConvB(32, 14, o1, e2, j, k, net->l2f[i], l2fm[i], net->l2fb[i], l2fbm[i]);
        }

        // convolve filter 3 with layer 3 error gradients
        for(uint i = 0; i < 128; i++) // num filter
        {
            for(uint j = 0; j < 7; j++) // height
                for(uint k = 0; k < 7; k++) // width
                    TBVGG3_3x3ConvB(64, 7, o2, e3, j, k, net->l3f[i], l3fm[i], net->l3fb[i], l3fbm[i]);
        }
        
        // weights nudged
    }

    // return activation
    return output;
}

#endif
