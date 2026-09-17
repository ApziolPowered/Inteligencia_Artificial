#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define epoca 3000
#define K 0.3f

//Funcion de Entrenamiento Perceptron
float EntNt(float, float, float);

//Funcion para las salidas
float InitNt(float, float);

//Sigmoide
float sigmoide(float);

//pesos aleatorios
void pesos_initNt();


//-----------------------------------------------------
// PESOS DE LA PRIMERA NEURONA
//-----------------------------------------------------

float Pesos[2];

float bias = 0.5f;


//-----------------------------------------------------
// PESOS DE LA SEGUNDA NEURONA
//-----------------------------------------------------

float Pesos2[2];

float bias2 = 0.5f;


//-----------------------------------------------------
// PESOS DE LA NEURONA DE SALIDA
//
// Esta neurona ya no recibe directamente x0 y x1.
//
// Recibe:
//
//      out
//      out2
//
// Es decir, las salidas de las dos neuronas anteriores.
//-----------------------------------------------------

float Pesos3[2];

float bias3 = 0.5f;


float Error;


//                  XOR
//
//                 1          1          0
//
//                 1          0          1
//
//                 0          1          1
//
//                 0          0          0
//


//-----------------------------------------------------
// FUNCION DE ENTRENAMIENTO
//-----------------------------------------------------

float EntNt(float x0, float x1, float target)
{

    //-------------------------------------------------
    // PRIMERA NEURONA
    //-------------------------------------------------

    float net = 0;

    float out = 0;

    float delta[2];
    //Es la variacion de los pesos sinapticos



    //-------------------------------------------------
    // SEGUNDA NEURONA
    //
    // Tiene sus propias variables.
    //
    // net2   = suma interna
    // out2   = salida de la neurona
    // delta2 = variacion de sus propios pesos
    //-------------------------------------------------

    float net2 = 0;

    float out2 = 0;

    float delta2[2];



    //-------------------------------------------------
    // TERCERA NEURONA
    //
    // Esta es la neurona de salida.
    //
    // Recibe:
    //
    // out  -> salida de la primera neurona
    // out2 -> salida de la segunda neurona
    //-------------------------------------------------

    float net3 = 0;

    float out3 = 0;

    float delta3[2];



    //-------------------------------------------------
    // DELTAS DE ERROR DE LAS NEURONAS
    //
    // Estos NO son los pesos.
    //
    // Representan cuanto debe corregirse cada neurona
    // debido al error cometido.
    //-------------------------------------------------

    float errorSalida;

    float errorN1;

    float errorN2;



    //-------------------------------------------------
    // FORWARD
    //-------------------------------------------------


    //-------------------------------------------------
    // PRIMERA NEURONA
    //-------------------------------------------------

    net = Pesos[0] * x0 +
          Pesos[1] * x1 -
          bias;

    net = sigmoide(net);

    out = net;



    //-------------------------------------------------
    // SEGUNDA NEURONA
    //
    // Utiliza LAS MISMAS entradas x0 y x1,
    // pero tiene pesos diferentes.
    //-------------------------------------------------

    net2 = Pesos2[0] * x0 +
           Pesos2[1] * x1 -
           bias2;

    net2 = sigmoide(net2);

    out2 = net2;



    //-------------------------------------------------
    // TERCERA NEURONA
    //
    // Aqui las entradas YA NO SON x0 y x1.
    //
    // Sus entradas son:
    //
    // out  = salida neurona 1
    // out2 = salida neurona 2
    //-------------------------------------------------

    net3 = Pesos3[0] * out +
           Pesos3[1] * out2 -
           bias3;

    net3 = sigmoide(net3);

    out3 = net3;



    //-------------------------------------------------
    // CALCULAR ERROR FINAL
    //-------------------------------------------------

    Error = target - out3;


    //printf("Error funcion %f \n", Error);



    //-------------------------------------------------
    // BACKPROPAGATION
    //-------------------------------------------------


    //-------------------------------------------------
    // ERROR DE LA NEURONA DE SALIDA
    //
    // La derivada de la sigmoide se puede obtener
    // usando:
    //
    // salida * (1 - salida)
    //-------------------------------------------------

    errorSalida = Error * out3 * (1 - out3);



    //-------------------------------------------------
    // ERROR QUE REGRESA A LA PRIMERA NEURONA
    //
    // La primera neurona contribuyo al resultado
    // mediante Pesos3[0].
    //
    // Por eso el error se multiplica por ese peso.
    //-------------------------------------------------

    errorN1 = errorSalida *
              Pesos3[0] *
              out *
              (1 - out);



    //-------------------------------------------------
    // ERROR QUE REGRESA A LA SEGUNDA NEURONA
    //-------------------------------------------------

    errorN2 = errorSalida *
              Pesos3[1] *
              out2 *
              (1 - out2);



    //-------------------------------------------------
    // ACTUALIZAR NEURONA DE SALIDA
    //-------------------------------------------------


    delta3[0] = K * errorSalida * out;

    delta3[1] = K * errorSalida * out2;


    Pesos3[0] += delta3[0];

    Pesos3[1] += delta3[1];


    //Como el bias es siempre 1,
    //el bias incluye ya su peso sinaptico

    bias3 -= K * errorSalida;



    //-------------------------------------------------
    // ACTUALIZAR PRIMERA NEURONA
    //-------------------------------------------------


    delta[0] = K * errorN1 * x0;
    //la variacion de los pesos sinapticos corresponde

    delta[1] = K * errorN1 * x1;
    //al error cometido, por la entrada correspondiente


    Pesos[0] += delta[0];
    //Se ajustan los nuevos valores

    Pesos[1] += delta[1];
    //de los pesos sinapticos


    bias -= K * errorN1;



    //-------------------------------------------------
    // ACTUALIZAR SEGUNDA NEURONA
    //
    // Exactamente igual que la primera,
    // pero utilizando SU propio error y SUS pesos.
    //-------------------------------------------------


    delta2[0] = K * errorN2 * x0;

    delta2[1] = K * errorN2 * x1;


    Pesos2[0] += delta2[0];

    Pesos2[1] += delta2[1];


    bias2 -= K * errorN2;



    //-------------------------------------------------
    // La salida final es la salida de la tercera
    // neurona.
    //-------------------------------------------------

    return out3;
}





//-----------------------------------------------------
// FUNCION PARA PROBAR LA RED
//
// Aqui NO se modifican los pesos.
// Solamente hacemos el FORWARD.
//-----------------------------------------------------

float InitNt(float x0, float x1)
{

    float net = 0;

    float out = 0;


    float net2 = 0;

    float out2 = 0;


    float net3 = 0;

    float out3 = 0;



    //-------------------------------------------------
    // PRIMERA NEURONA
    //-------------------------------------------------

    net = Pesos[0] * x0 +
          Pesos[1] * x1 -
          bias;

    net = sigmoide(net);

    out = net;



    //-------------------------------------------------
    // SEGUNDA NEURONA
    //-------------------------------------------------

    net2 = Pesos2[0] * x0 +
           Pesos2[1] * x1 -
           bias2;

    net2 = sigmoide(net2);

    out2 = net2;



    //-------------------------------------------------
    // TERCERA NEURONA
    //-------------------------------------------------

    net3 = Pesos3[0] * out +
           Pesos3[1] * out2 -
           bias3;

    net3 = sigmoide(net3);

    out3 = net3;


    return out3;
}





//-----------------------------------------------------
// INICIALIZAR PESOS ALEATORIOS
//-----------------------------------------------------

void pesos_initNt(void)
{

    int i;


    //-------------------------------------------------
    // PESOS DE LA PRIMERA NEURONA
    //-------------------------------------------------

    for(i = 0; i < 2; i++)
    {

        Pesos[i] = (float)rand() / RAND_MAX;

    }



    //-------------------------------------------------
    // PESOS DE LA SEGUNDA NEURONA
    //-------------------------------------------------

    for(i = 0; i < 2; i++)
    {

        Pesos2[i] = (float)rand() / RAND_MAX;

    }



    //-------------------------------------------------
    // PESOS DE LA NEURONA DE SALIDA
    //-------------------------------------------------

    for(i = 0; i < 2; i++)
    {

        Pesos3[i] = (float)rand() / RAND_MAX;

    }

}





//-----------------------------------------------------
// SIGMOIDE
//-----------------------------------------------------

float sigmoide(float s)
{

    return 1.0f / (1.0f + expf(-s));

}





int main()
{

    int i = 0;

    float apr;


    pesos_initNt();



    while(i < epoca)
    {

        printf("------------------------\n");

        printf("Salida Entrenamiento Epoca %d \n", i);



        //-------------------------------------------------
        // TABLA DE VERDAD XOR
        //-------------------------------------------------


        apr = EntNt(1, 1, 0);

        printf("1,1 = %f\n", apr);



        apr = EntNt(1, 0, 1);

        printf("1,0 = %f\n", apr);



        apr = EntNt(0, 1, 1);

        printf("0,1 = %f\n", apr);



        apr = EntNt(0, 0, 0);

        printf("0,0 = %f\n", apr);



        printf("\n");



        //-------------------------------------------------
        // PESOS DE LA PRIMERA NEURONA
        //-------------------------------------------------

        printf("Pesos Neurona 1\n");

        printf("Peso 0 = %f\n", Pesos[0]);

        printf("Peso 1 = %f\n", Pesos[1]);

        printf("Bias = %f\n", bias);



        //-------------------------------------------------
        // PESOS DE LA SEGUNDA NEURONA
        //-------------------------------------------------

        printf("\nPesos Neurona 2\n");

        printf("Peso 0 = %f\n", Pesos2[0]);

        printf("Peso 1 = %f\n", Pesos2[1]);

        printf("Bias = %f\n", bias2);



        //-------------------------------------------------
        // PESOS DE LA NEURONA DE SALIDA
        //-------------------------------------------------

        printf("\nPesos Neurona Salida\n");

        printf("Peso 0 = %f\n", Pesos3[0]);

        printf("Peso 1 = %f\n", Pesos3[1]);

        printf("Bias = %f\n", bias3);



        printf("\nError %f\n", Error);


        printf("------------------------\n");


        i++;

    }



    //-------------------------------------------------
    // RESULTADOS FINALES
    //-------------------------------------------------

    printf("\n\n");

    printf("============================\n");

    printf("RESULTADOS FINALES XOR\n");

    printf("============================\n");



    apr = InitNt(0, 0);

    printf("0 XOR 0 = %f -> %d\n",
           apr,
           apr >= 0.5f ? 1 : 0);



    apr = InitNt(0, 1);

    printf("0 XOR 1 = %f -> %d\n",
           apr,
           apr >= 0.5f ? 1 : 0);



    apr = InitNt(1, 0);

    printf("1 XOR 0 = %f -> %d\n",
           apr,
           apr >= 0.5f ? 1 : 0);



    apr = InitNt(1, 1);

    printf("1 XOR 1 = %f -> %d\n",
           apr,
           apr >= 0.5f ? 1 : 0);

printf("\nPresiona ENTER para salir...");
getchar();

    return 0;
}