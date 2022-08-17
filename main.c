// James William Fletcher (github.com/mrbid)
// github.com/TFCNN/TBVGG3
// gcc main.c -lm -Ofast -mavx -mfma -o main

#include <signal.h>
#define forceinline __attribute__((always_inline)) inline

//#define LINUX_DEBUG
//#define UNIFORM_GLOROT
#define ADA16
#define LEARNING_RATE 0.0001f
#define GAIN          0.0065f
#define SIGMOID_OUTPUT
#include "TBVGG3_ADA.h"
#define NORMALISE_INPUTS
#define TRAIN_NONTARGETS
#define NONTARGETS_ZERO // strangely training the network to output zero with zero'd input data improves the accuracy O_o

/*
    In my first ever TBVGG3 implementation on the first CS:GO dataset
    I mean normalised every image by the standard deviation of the
    entire dataset, not by mini batches.
    
    There are a few main arguments to how one should normalise image data:
    1. stddev of dataset
    2. stddev of mini batch
    3. stddev per image
    4. 0-1 scaling per pixel
    5. -1 to +1 scaling per pixel
    
    What you choose, ultimately depends on your dataset, but don't
    overcomplicate it in your first attempt, method 4 or 5 is a good
    start and if that goes well you *might* be able to squeeze out
    more accuracy with stddev normalisation.
    
    Working your way starting from 5 to 1 is probably your best bet
    to find the best solution for your dataset.
*/

#define MAX_SAMPLES 5179 // we use NONTARGET_SAMPLES as it's the smaller of the two counts
#define TARGET_SAMPLES MAX_SAMPLES // 5216
#define NONTARGET_SAMPLES MAX_SAMPLES // 5179
#define SAMPLE_SIZE 2352
#define EPOCHS 333333333

float targets[TARGET_SAMPLES][3][28][28] = {0};
#ifdef TRAIN_NONTARGETS
    float nontargets[NONTARGET_SAMPLES][3][28][28] = {0};
#endif
TBVGG3_Network net;

///

static forceinline uint uRand(const uint min, const uint max)
{
    static float rndmax = 1.f/(float)RAND_MAX;
    return ((((float)rand()) * rndmax) * ((max+1)-min) ) + min;
}

int srandfq = 1988;
forceinline float urandfc() // -1 to 1
{
    // https://www.musicdsp.org/en/latest/Other/273-fast-f32-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return ((float)srandfq) * 4.6566129e-010f;
}

void shuffle_dataset()
{
    const int dl = SAMPLE_SIZE*sizeof(float);
    const int DS1 = MAX_SAMPLES-1;
    
    for(int i = 0; i < MAX_SAMPLES; i++)
    {
        const int i1 = uRand(0, DS1);
        int i2 = i1;
        while(i1 == i2)
            i2 = uRand(0, DS1);
        float t[SAMPLE_SIZE];
        memcpy(&t, &targets[i1], dl);
        memcpy(&targets[i1], &targets[i2], dl);
        memcpy(&targets[i2], &t, dl);
    }

#ifdef TRAIN_NONTARGETS
    for(int i = 0; i < MAX_SAMPLES; i++)
    {
        const int i1 = uRand(0, DS1);
        int i2 = i1;
        while(i1 == i2)
            i2 = uRand(0, DS1);
        float t[SAMPLE_SIZE];
        memcpy(&t, &nontargets[i1], dl);
        memcpy(&nontargets[i1], &nontargets[i2], dl);
        memcpy(&nontargets[i2], &t, dl);
    }
#endif
}

void generate_output(int sig_num)
{
    // save network
    TBVGG3_SaveNetwork(&net, "network.save");

#ifdef LINUX_DEBUG
    // dump weights
    char fn[256];
    sprintf(fn, "debug/final");
    TBVGG3_Dump(&net, fn);
#endif

    // done
    exit(0);
}

int main()
{
    // ctrl+c callback
    signal(SIGINT, generate_output);

    // seed random
    srand(1988);

    // load targets
    printf("loading target data\n");
    for(int i = 0; i < TARGET_SAMPLES; i++)
    {
        char fn[256];
        sprintf(fn, "target/%i.ppm", i+1);
        FILE* f = fopen(fn, "rb");
        if(f)
        {
            // notify load
            printf("%s\n", fn);

            // seek past ppm header
            fseek(f, 13, SEEK_SET);

            // read bytes into temp buffer
            unsigned char tb[SAMPLE_SIZE];
            if(fread(&tb, 1, SAMPLE_SIZE, f) !=  SAMPLE_SIZE)
            {
                printf("read error\n");
                return 0;
            }

            // cast byte array to floats & normalise -1 to 1
            for(int y = 0; y < 28; y++)
            {
                for(int x = 0; x < 28; x++)
                {
                    const float r = (float)tb[(((28*y)+x)*3)];
                    const float g = (float)tb[(((28*y)+x)*3)+1];
                    const float b = (float)tb[(((28*y)+x)*3)+2];
#ifdef NORMALISE_INPUTS
                    targets[i][0][x][y] = (r-128.f)*0.003921568859f;
                    targets[i][1][x][y] = (g-128.f)*0.003921568859f;
                    targets[i][2][x][y] = (b-128.f)*0.003921568859f;
#else
                    targets[i][0][x][y] = r;
                    targets[i][1][x][y] = g;
                    targets[i][2][x][y] = b;
#endif
                }
            }

            // done
            fclose(f);
        }
    }

#ifdef TRAIN_NONTARGETS
#ifndef NONTARGETS_ZERO
    // load nontargets
    printf("\nloading nontarget data\n");
    for(int i = 0; i < MAX_SAMPLES; i++)
    {
        char fn[256];
        sprintf(fn, "nontarget/%i.ppm", i+1);
        FILE* f = fopen(fn, "rb");
        if(f)
        {
            // notify load
            printf("%s\n", fn);

            // seek past ppm header
            fseek(f, 13, SEEK_SET);

            // read bytes into temp buffer
            unsigned char tb[SAMPLE_SIZE];
            if(fread(&tb, 1, SAMPLE_SIZE, f) !=  SAMPLE_SIZE)
            {
                printf("read error\n");
                return 0;
            }

            // cast byte array to floats & normalise -1 to 1
            for(int y = 0; y < 28; y++)
            {
                for(int x = 0; x < 28; x++)
                {
                    const float r = (float)tb[(((28*y)+x)*3)];
                    const float g = (float)tb[(((28*y)+x)*3)+1];
                    const float b = (float)tb[(((28*y)+x)*3)+2];
                    
                    // it should be C,Y,X not C,X,Y (colour channel, y-axis, x-axis)
                    // bit late to change it now, but if you are doing any custom projects
                    // do consider swapping the x and y terms around on the code below.
#ifdef NORMALISE_INPUTS
                    nontargets[i][0][x][y] = (r-128.f)*0.003921568859f; // swap x & y
                    nontargets[i][1][x][y] = (g-128.f)*0.003921568859f; // swap x & y
                    nontargets[i][2][x][y] = (b-128.f)*0.003921568859f; // swap x & y
#else
                    nontargets[i][0][x][y] = r; // swap x & y
                    nontargets[i][1][x][y] = g; // swap x & y
                    nontargets[i][2][x][y] = b; // swap x & y
#endif
                }
            }

            // done
            fclose(f);
        }
    }
#endif
#endif

    // train
    printf("\ntraining network\n\n");
    TBVGG3_Reset(&net, time(0));
    for(int i = 0; i < EPOCHS; i++)
    {
        float epoch_loss = 0.f;
        time_t st = time(0);
        for(int j = 0; j < MAX_SAMPLES; j++)
        {
            float r = 1.f - TBVGG3_Process(&net, targets[j], LEARN_MAX);
#ifdef TRAIN_NONTARGETS
            r += TBVGG3_Process(&net, nontargets[j], LEARN_MIN);
#endif
            epoch_loss += r;
            //printf("[%i] loss: %f\n", j, r);
        }
        shuffle_dataset();
        printf("[%i] epoch loss: %f\n", i, epoch_loss);
        printf("[%i] avg epoch loss: %f\n", i, epoch_loss/MAX_SAMPLES);
        TBVGG3_Debug(&net);
        printf("[%i] SPS: %.2f\n\n", i, ((float)MAX_SAMPLES)/((float)(time(0)-st))); // samples per second
    }

    // done
    generate_output(0);
    return 0;
}
