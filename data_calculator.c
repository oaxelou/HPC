#include <stdio.h>
#include <math.h>

int main(int argc,char *argvp[]){
    int i;
    double min, max;
    double timing[12], sum, mean;
    double std_dev;

    scanf("%lf", &timing[0]);
    min = timing[0];
    max = timing[0];
    for(i = 1; i < 12; i++){
        scanf("%lf", &timing[i]);
        if(timing[i] < min)
          min = timing[i];
        if(timing[i] > max)
          max = timing[i];
    }

    sum = 0;
    for(i = 0; i < 12; i++){
      if(timing[i] != min && timing[i] != max)
        sum += timing[i];
    }

    mean = sum / ((double) 10);
    std_dev = 0;
    for(i = 0; i < 12; i++){
      if(timing[i] != min && timing[i] != max)
        std_dev += (timing[i] - mean) * (timing[i] - mean);
    }

    std_dev /= 10;

    std_dev = sqrt(std_dev);
    printf("-------------------\nmean: %lf\n", mean);
    printf("standard deviation: %lf\n", std_dev);

    return 0;
}
