
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//Função que calcula a média de uma "matriz" 5x5 a partir de uma dada posição
__global__ void smooth( int *entrada,int *saida, int n_linhas, int n_colunas )
{
    //Calcula a posição no vetor (id_bloco * total_blocos + id_thread)
    int posicao = blockIdx.x * blockDim.x + threadIdx.x;
    //Se a posição não é maior que o limite da imagem original...
    if(posicao < (n_linhas)*(n_colunas))
    {
        //soma o valor da região 5x5 em torno no pixel
        saida[posicao] =entrada[posicao]+
                        entrada[posicao+(n_colunas+4)]+
                        entrada[posicao+(2*(n_colunas+4))]+
                        entrada[posicao+(3*(n_colunas+4))]+
                        entrada[posicao+(4*(n_colunas+4))]+
                        entrada[posicao+1]+
                        entrada[posicao+(n_colunas+4)+1]+
                        entrada[posicao+(2*(n_colunas+4))+1]+
                        entrada[posicao+(3*(n_colunas+4))+1]+
                        entrada[posicao+(4*(n_colunas+4))+1]+
                        entrada[posicao+2]+
                        entrada[posicao+(n_colunas+4)+2]+
                        entrada[posicao+(2*(n_colunas+4))+2]+
                        entrada[posicao+(3*(n_colunas+4))+2]+
                        entrada[posicao+(4*(n_colunas+4))+2]+
                        entrada[posicao+3]+
                        entrada[posicao+(n_colunas+4)+3]+
                        entrada[posicao+(2*(n_colunas+4))+3]+
                        entrada[posicao+(3*(n_colunas+4))+3]+
                        entrada[posicao+(4*(n_colunas+4))+3]+
                        entrada[posicao+4]+
                        entrada[posicao+(n_colunas+4)+4]+
                        entrada[posicao+(2*(n_colunas+4))+4]+
                        entrada[posicao+(3*(n_colunas+4))+4]+
                        entrada[posicao+(4*(n_colunas+4))+4];
        //calcula a média
        saida[posicao] = saida[posicao]/25;
    }
}

int main( void )
{
    //tamanho da imagem
    int n_linhas, n_colunas;

    //valor limite declarado no arquivo RBG (neste caso 255)
    int limite_RGB;

    //variaveis auxiliares
    int i,j;

    //primeira linha do aquivo .ppm (P3)
    char line1[5];

    //hashtag
    char hashTag[20];

    //le da entrada padrão o cabeçalho do arquivo .ppm
    scanf("%s",line1);
    scanf("%s",hashTag);
    scanf("%d",&n_colunas);
    scanf("%d",&n_linhas);
    scanf("%d",&limite_RGB);

    //numero maximo de threads da placa do andromeda
    int nthreads = 1024;

    float nb;
    int numBlocks;

    //numero de blocos é o total de pixels dividido pelo total de threads
    nb = (n_colunas*n_linhas)/nthreads;

    //O cast trunca o ponto flutuante, por isso soma-se 1
    numBlocks = (int) (nb + 1.0);

    //Aloca um vetor para cada componente de cor com o total de pixels da imagem
    int *vermelho_original = (int*)calloc((n_linhas+4)*(n_colunas+4),sizeof(int));
    int *vermelho_final = (int*)calloc((n_linhas)*(n_colunas),sizeof(int));
    int *verde_original = (int*)calloc((n_linhas+4)*(n_colunas+4),sizeof(int));
    int *verde_final = (int*)calloc((n_linhas)*(n_colunas),sizeof(int));
    int *azul_original = (int*)calloc((n_linhas+4)*(n_colunas+4),sizeof(int));
    int *azul_final = (int*)calloc((n_linhas)*(n_colunas),sizeof(int));

    //le cada componente de cor da imagem e salva nos verotes originais
    for(i=0; i<n_linhas; i++)
    {
        for(j=0; j<n_colunas; j++)
        {
            scanf("%d",&vermelho_original[((i+2)*(n_colunas+4))+j+2]);
            scanf("%d",&verde_original[((i+2)*(n_colunas+4))+j+2]);
            scanf("%d",&azul_original[((i+2)*(n_colunas+4))+j+2]);
        }
    }

    int *original,*final;

    //Malloc especial do CUDA, para os vetores originais e de saída
    //Estes vetores são passados às funções que serão calculadas pela
    //placa de vídeo
    cudaMalloc( (void**)&original, (n_linhas+4)*(n_colunas+4) * sizeof(int));
    cudaMalloc( (void**)&final, (n_linhas)*(n_colunas) * sizeof(int));

    //Calcula a partir deste ponto o tempo que foi preciso para fazer os cálculos
    struct timeval inicio, fim;
    gettimeofday(&inicio,0);


    //VERMELHO (R)
    //Fução do CUDA que copia o conteudo do vetor lido para o vetor a ser passado para a função
    cudaMemcpy( original, vermelho_original, (n_linhas+4)*(n_colunas+4)  * sizeof(int),cudaMemcpyHostToDevice);
    //Função que faz o smooth para cada thread em cada bloco
    smooth<<<numBlocks,nthreads>>>(original,final,n_linhas,n_colunas);
    //Função do CUDA que copia o conteudo do vetor que foi calculado na função acima para o vetor a ser gravado no arquivo final
    cudaMemcpy(vermelho_final, final, (n_linhas)*(n_colunas) * sizeof(int),cudaMemcpyDeviceToHost);


    //VERDE (G)
    //Fução do CUDA que copia o conteudo do vetor lido para o vetor a ser passado para a função
    cudaMemcpy( original, verde_original, (n_linhas+4)*(n_colunas+4) * sizeof(int),cudaMemcpyHostToDevice);
    //Função que faz o smooth para cada thread em cada bloco
    smooth<<<numBlocks,nthreads>>>(original,final,n_linhas,n_colunas);
    //Função do CUDA que copia o conteudo do vetor que foi calculado na função acima para o vetor a ser gravado no arquivo final
    cudaMemcpy(verde_final, final, (n_linhas)*(n_colunas) * sizeof(int),cudaMemcpyDeviceToHost);


    //AZUL (B)
    //Fução do CUDA que copia o conteudo do vetor lido para o vetor a ser passado para a função
    cudaMemcpy( original, azul_original, (n_linhas+4)*(n_colunas+4) * sizeof(int),cudaMemcpyHostToDevice);
    //Função que faz o smooth para cada thread em cada bloco
    smooth<<<numBlocks,nthreads>>>(original,final,n_linhas,n_colunas);
    //Função do CUDA que copia o conteudo do vetor que foi calculado na função acima para o vetor a ser gravado no arquivo final
    cudaMemcpy(azul_final, final, (n_linhas)*(n_colunas) * sizeof(int),cudaMemcpyDeviceToHost);

    //Calcula até este ponto o tempo que foi preciso para fazer os cálculos
    gettimeofday(&fim,0);
    float speedup = (fim.tv_sec + fim.tv_usec/1000000.0) - (inicio.tv_sec + inicio.tv_usec/1000000.0);
    printf("tempo: %f\n", speedup);

    //escreve no arquivo de saida
    FILE *arqOut = fopen("out.ppm","w+");
    fprintf(arqOut,"%s\n",line1);
    fprintf(arqOut,"%s\n",hashTag);
    fprintf(arqOut,"%d %d\n",n_colunas,n_linhas);
    fprintf(arqOut,"%d \n",limite_RGB);
    for(i=0; i<n_linhas; i++)
    {
        for(j=0; j<n_colunas; j++)
        {
            fprintf(arqOut,"%d ",vermelho_final[(i*(n_colunas))+j]);
            fprintf(arqOut,"%d ",verde_final[(i*(n_colunas))+j]);
            fprintf(arqOut,"%d ",azul_final[(i*(n_colunas))+j]);
        }
        fprintf(arqOut,"\n");
    }
    fclose(arqOut);

    //libera os vetores utilizados
    free(vermelho_original);
    free(verde_original);
    free(azul_original);
    free(vermelho_final);
    free(verde_final);
    free(azul_final);
    cudaFree(original);
    cudaFree(final);

    return(0);
}
