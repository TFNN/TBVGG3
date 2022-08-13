// James William Fletcher (github.com/mrbid)
// github.com/TFCNN/TBVGG3
// gcc main.c -lm -Ofast -mavx -mfma -o main

#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>

#include <sys/file.h>

#define LINUX_DEBUG
#include "TBVGG3_ADA_MED.h"

#define uint unsigned int
#define forceinline __attribute__((always_inline)) inline

#define MAX_SAMPLES 5179 // we use NONTARGET_SAMPLES as it's the smaller count
#define TARGET_SAMPLES MAX_SAMPLES // 5216
#define NONTARGET_SAMPLES MAX_SAMPLES // 5179
#define SAMPLE_SIZE 2352
#define EPOCHS 333

float targets[TARGET_SAMPLES][3][28][28];
float nontargets[NONTARGET_SAMPLES][3][28][28];
TBVGG3_Network net;

///
///// ----------
#define FAST_PREDICTABLE_MODE

int srandfq = 1988;
void srandf(const int seed)
{
    srandfq = seed;
}

forceinline float urandf() // 0 to 1
{
#ifdef FAST_PREDICTABLE_MODE
    // https://www.musicdsp.org/en/latest/Other/273-fast-float-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return (float)(srandfq & 0x7FFFFFFF) * 4.6566129e-010f;
#else
    static const float FLOAT_UINT64_MAX = (float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s)+1e-7f) / FLOAT_UINT64_MAX;
#endif
}

forceinline float uRandFloat(const float min, const float max)
{
    return ( urandf() * (max-min) ) + min;
}

forceinline unsigned int uRand(const uint min, const uint umax)
{
    const uint max = umax + 1;
    return ( urandf() * (max-min) ) + min;
}
///// ----------
///

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

    // dump weights
    char fn[256];
    sprintf(fn, "debug/final");
    TBVGG3_Dump(&net, fn);

    // done
    exit(0);
}

int main()
{
    // ctrl+c callback
    signal(SIGINT, generate_output);

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

            // cast byte array to floats
            for(int y = 0; y < 28; y++)
            {
                for(int x = 0; x < 28; x++)
                {
                    const float r = (float)tb[(((28*y)+x)*3)];
                    const float g = (float)tb[(((28*y)+x)*3)+1];
                    const float b = (float)tb[(((28*y)+x)*3)+2];
                    targets[i][0][x][y] = r;
                    targets[i][1][x][y] = g;
                    targets[i][2][x][y] = b;
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

            // cast byte array to floats
            for(int y = 0; y < 28; y++)
            {
                for(int x = 0; x < 28; x++)
                {
                    const float r = (float)tb[(28*y)+x];
                    const float g = (float)tb[((28*y)+x)+1];
                    const float b = (float)tb[((28*y)+x)+2];
                    nontargets[i][0][x][y] = r;
                    nontargets[i][1][x][y] = g;
                    nontargets[i][2][x][y] = b;
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
        for(int j = 0; j < MAX_SAMPLES; j++)
        {
            float r = 1.f - TBVGG3_Process(&net, targets[j], LEARN_MAX);
            r += TBVGG3_Process(&net, nontargets[j], LEARN_MIN);
            epoch_loss += r;
            //printf("[%i] loss: %.3f\n", j, r);
        }
        shuffle_dataset();
        printf("[%i] epoch loss: %.3f\n", i, epoch_loss);
        printf("[%i] avg epoch loss: %.3f\n\n", i, epoch_loss/MAX_SAMPLES);
    }

    // done
    generate_output(0);
    return 0;
}
