// James William Fletcher (github.com/mrbid)
// github.com/TFCNN/TBVGG3
// gcc main.c -lm -Ofast -mavx -mfma -o main

#include <signal.h>
#define forceinline __attribute__((always_inline)) inline

//#define LINUX_DEBUG
//#define UNIFORM_GLOROT
#define SIGMOID_OUTPUT
#include "TBVGG3_ADA_MED.h"
#define NORMALISE_INPUTS

#define MAX_SAMPLES 5179 // we use NONTARGET_SAMPLES as it's the smaller count
#define TARGET_SAMPLES MAX_SAMPLES // 5216
#define NONTARGET_SAMPLES MAX_SAMPLES // 5179
#define SAMPLE_SIZE 2352
#define EPOCHS 333

float targets[TARGET_SAMPLES][3][28][28];
float nontargets[NONTARGET_SAMPLES][3][28][28];
TBVGG3_Network net;

///

static forceinline uint uRand(const uint min, const uint max)
{
    static float rndmax = 1.f/(float)RAND_MAX;
    return (((((float)rand())+1e-7f) * rndmax) * ((max+1)-min) ) + min;
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
        printf("%s\n", fn);
        FILE* f = fopen(fn, "rb");
        if(f)
        {
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

    // load nontargets
    printf("\nloading nontarget data\n");
    for(int i = 0; i < MAX_SAMPLES; i++)
    {
        char fn[256];
        sprintf(fn, "nontargets/%i.ppm", i+1);
        printf("%s\n", fn);
        FILE* f = fopen(fn, "rb");
        if(f)
        {
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
                    const float r = (float)tb[(28*y)+x];
                    const float g = (float)tb[((28*y)+x)+1];
                    const float b = (float)tb[((28*y)+x)+2];
#ifdef NORMALISE_INPUTS
                    nontargets[i][0][x][y] = (r-128.f)*0.003921568859f;
                    nontargets[i][1][x][y] = (g-128.f)*0.003921568859f;
                    nontargets[i][2][x][y] = (b-128.f)*0.003921568859f;
#else
                    nontargets[i][0][x][y] = r;
                    nontargets[i][1][x][y] = g;
                    nontargets[i][2][x][y] = b;
#endif
                }
            }

            // done
            fclose(f);
        }
    }

    // train
    printf("\ntraining network\n\n");
    TBVGG3_Reset(&net);
    for(int i = 0; i < EPOCHS; i++)
    {
        float epoch_loss = 0.f;
        time_t st = time(0);
        for(int j = 0; j < MAX_SAMPLES; j++)
        {
            float r = 1.f - TBVGG3_Process(&net, targets[j], LEARN_MAX);
            r += TBVGG3_Process(&net, nontargets[j], LEARN_MIN);
            epoch_loss += r;
            //printf("[%i] loss: %f\n", j, r);
        }
        shuffle_dataset();
        printf("[%i] epoch loss: %f\n", i, epoch_loss);
        printf("[%i] avg epoch loss: %f\n", i, epoch_loss/MAX_SAMPLES);
        printf("[%i] SPS: %.2f\n\n", i, ((float)MAX_SAMPLES)/((float)(time(0)-st))); // samples per second
    }

    // done
    generate_output(0);
    return 0;
}
