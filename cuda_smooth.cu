/*SmoothSeq SmoothSeq.cpp `pkg-config --cflags --libs opencv`
    ./SmoothSeq image_in type_img image_out
    type_img -> 0 = GRAYSCALE
    type_img -> 1 = COLOR
*/
    
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <math.h>

using namespace cv;
using namespace std;

//Função que calcula a média de uma "matriz" 5x5 a partir de uma dada posição
__global__ void applyMask(unsigned char* m_in, unsigned char* m_out, int height, int width) {

        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        int lin = tid/(width);
        int col = tid%(width);
        int h = (height);
        int w = (width);
        int i, j;
        int sum;

        /* Doesn't do anything for the borders */
        if (tid < w * h){
                if (lin >= 2 && col >= 2 && lin <= h-3 && col <= w-3){
                        sum = 0;
                        for (i=-2; i<=2; i++){
                                for (j=-2; j<=2; j++){
                                        sum += m_in[tid + i*w + j];
                                }
                        }
                        m_out[tid] = sum/25;
                }
                else {
                        m_out[tid] = m_in[tid];
                }
        }
}


int main(int argc, char *argv[]) {
    //diz se a imagem é grayscale or color
    int tipo_img = atoi(argv[2]);
    //arquivo de entrada
    const char *fileIn, *fileOut;
    
    //numero maximo de threads da placa do andromeda
    int nthreads = 1024;

    int numBlocks;


    //matriz com a imagem de entrada
    Mat in;
    //matriz que receberá a imagem de saida
    Mat out;

    //le o nome da imagem
    fileIn = argv[1];
    fileOut = argv[3];
    //le e salva a imagem na matriz
    if(tipo_img == 0) {
        in = imread(fileIn, CV_LOAD_IMAGE_GRAYSCALE);
    } else if(tipo_img == 1) {
        in = imread(fileIn, CV_LOAD_IMAGE_COLOR);
    } else {
        cout << "Tipo de imagem nao suportado" << endl;
        return -1;
    }
    //caso nao consegui abrir a imagem
    if (in.empty()) {
        cout << "Nao foi possivel abrir a  imagem: " << endl;
        return -1;
    }
    int border = 2;

    int l_height = in.size().height, l_width = in.size().width;

    //numero de blocos é o total de pixels dividido pelo total de threads
    numBlocks = (l_height*l_width/nthreads)+1;

    unsigned char *original,*saida;
  
    //Malloc especial do CUDA, para os vetores originais e de saída
    //Estes vetores são passados às funções que serão calculadas pela
    //placa de vídeo
    
    copyMakeBorder(in, in, border, border, border, border, BORDER_REPLICATE);
    cudaMalloc(&original, (l_width + 4) * (l_height + 4));
    cudaMalloc(&saida, l_width * l_height);

    //pegar o tempo de inicio
    out = Mat::zeros(int.size (), in.type());

    struct timeval inicio, fim;
    gettimeofday(&inicio,0);
    
    cudaMemcpy(original, in.data,(l_width + 4) * (l_height + 4), cudaMemcpyHostToDevice);

    smooth<<<numBlocks,nthreads>>>(original, saida, l_height, l_width);
    
    cudaMemcpy(out.data, saida, l_width*l_height,cudaMemcpyDeviceToHost);
    

    //pega o tempo de fim, faz a diferença e imprime na tela
    gettimeofday(&fim,0);
    float speedup = (fim.tv_sec + fim.tv_usec/1000000.0) - (inicio.tv_sec + inicio.tv_usec/1000000.0);
    cout << speedup << endl;
    imwrite(fileOut, out);
    in.release();
    out.release();
    cudaFree(original);
    cudaFree(saida);

    return 0;
}
    