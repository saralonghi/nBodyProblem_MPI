#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>

#define DT 0.01
#define MASTER 0
#define SOFTENING 1e-9 

#define EXPECTED_ARGUMENT 4
#define ARG1_BODIES 1
#define ARG2_ITERATIONS 2
#define ARG3_READ 3

#define CORRECTLY_INVOKED 1
#define NOT_CORRECTLY_INVOKED 0

// Body structure definition
typedef struct { 
    double m; // in kg  
    double x; // per 10.000 pianeti 2e9 --->	Lato del piano = 2𝑒7 × SQRT 𝑁 km (con N num di body)
    double y; // per 10.000 pianeti 2e9
    double vx; // in km/s 
    double vy; // in Km/s 
} Body;

const double G = 6.67430e-11;  // m^3·kg⁻¹·s⁻²

// Functions definition
void randomizeBodies(Body *bodies, int numberOfBodies);
void bodyForce(Body *localBodies1, Body *bodies2, int localBodies1Length,int bodies2Length, double *sumDX, double *sumDY, double *sumVX, double *sumVY);
void computeForce(Body BodyA, Body BodyB, double *sum_dx, double *sum_dy,double *sum_vx, double *sum_vy);
void updatePositions(Body *localBodies, int localBodiesLength, double *sum_dx, double *sum_dy, double *sum_vx, double *sum_vy);
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs);
void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, double executionTime, int isExecutionTimeRequired);
void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd);
void printHowToUse();
void readBodies( Body *bodies, int numberOfBodies);


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int numberOfTasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int isCorrectlyInvoked = CORRECTLY_INVOKED;
    if (argc != EXPECTED_ARGUMENT) {
        isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;
    }

    int reading = 0; 
    if (isCorrectlyInvoked != NOT_CORRECTLY_INVOKED) {  
        if (strcmp(argv[ARG3_READ], "-R") == 0) {
            reading = 1;
        } else if (strcmp(argv[ARG3_READ], "-G") == 0) {
            reading = 0;
        } else {
            isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;
        }
    }
    if (isCorrectlyInvoked == NOT_CORRECTLY_INVOKED) {
        if (rank == MASTER) {
            printHowToUse();
        }
        MPI_Finalize();
        return 0;
    }

   
    int numberOfBodies = atoi(argv[ARG1_BODIES]);
    int iterations = atoi(argv[ARG2_ITERATIONS]);
    srand(53);

    int bytes = numberOfBodies * sizeof(Body);
    Body *bodies = NULL;

    /* Se sono nel processo 0 creo un array di float "buffer" e un array di 
    body "bodies" e poi chiamo la funzione per inizializzare i float*/
   if (rank == MASTER) {
    bodies = (Body *) malloc(bytes);  // Qui NON usare "Body *" per non ridefinire la variabile!
    
    if(reading == 0){
        randomizeBodies(bodies, numberOfBodies);
    }else{
        readBodies(bodies, numberOfBodies);
    }
    printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations, 0 , 0);
}

    // --- Definizione del tipo MPI custom
    /* SPIEGO PERCHE' BISOGNA COSTRUIRE UNA FUNZIONE MPI E COSA è L'OFFSET E IL PADDING 
       struct Example { char a;  double b; }; //char = 1byte, double = 8byte
       Sembra che questa struct occupi 1 + 8 = 9 byte, giusto?  In realtà occupa 16 byte.    
       "char a" occupa il primo byte. Poi ci sono 7 byte di padding per allineare "double b" a
       un indirizzo multiplo di 8 e b occupa gli 8 byte successivi.

       Byte offset:   0  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
                     [a] [-padding-] [----------b----------]
    */
    MPI_Datatype MPI_BODY;
    int blocksCount[5] = {1, 1, 1, 1, 1};
    MPI_Datatype oldTypes[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[5];

    offsets[0] = offsetof(Body, m);
    offsets[1] = offsetof(Body, x);
    offsets[2] = offsetof(Body, y);
    offsets[3] = offsetof(Body, vx); 
    offsets[4] = offsetof(Body, vy);

    MPI_Type_create_struct(5, blocksCount, offsets, oldTypes, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
 

    /* numberOfTask: è il numero di processi tot 
     inizializzo bodiesPerProcess e displs entrembi come un array di int di grandezza numberOfTask 
     bodiesPerProcess/displs = int arr[processi tot]*/
    int *bodiesPerProcess = (int*) malloc(numberOfTasks * sizeof(int)); 
    int *displs           = (int*) malloc(numberOfTasks * sizeof(int)); 
    
  
    buildBodiesPerProcessAndDispls(numberOfBodies, numberOfTasks, bodiesPerProcess, displs);

    MPI_Bcast(bodiesPerProcess, numberOfTasks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, numberOfTasks, MPI_INT, 0, MPI_COMM_WORLD);

    //In MPI, ogni processo ha il proprio spazio di memoria separato (modello distributed memory).
    /*Viene allocato il numero di corpi che il processo specificato da rank deve gestire
    (Body*) serve a convertire (fare il cast) del risultato di malloc al tipo corretto, cioè Body*. */
    Body *localBodies = (Body*) malloc(bodiesPerProcess[rank] * sizeof(Body));
     
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizza tutti i clock dei processi
    double startTime = MPI_Wtime(); 

    // Scatterv: master invia, ogni processo riceve in buffer locale
    MPI_Scatterv(
        bodies, bodiesPerProcess, displs, MPI_BODY,
        localBodies, bodiesPerProcess[rank], MPI_BODY,
        MASTER, MPI_COMM_WORLD
    );


/*                  0     2       5     7
        int data[10] = {1,2,  3,4,5,  6,7,  8,9,10};
        int sendcounts[4] = {2, 3, 2, 3};      //quanti elementi riceve ogni processo
        int displs[4]     = {0, 2, 5, 7};      //dove inizia il segmento per ogni processo
    */

    double *sum_dx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   // sum_dx[3]={ DX-D, DX-E, DX-F }
    double *sum_dy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   // sum_dy[3]={ DY-D, DY-E,DY-F }
    double *sum_vx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   // sum_dx[3]={ DX-D, DX-E, DX-F }
    double *sum_vy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   // sum_dy[3]={ DY-D, DY-E,DY-F }
		

    /*TODO: se questo pezzo 160-164 non ci fosse ? non è superfluo comunicare i blocchi di bodies che ogni processo ha preso?
     È essenziale per sincronizzare le porzioni di dati tra i processi.
     Anche se ogni processo sa quanti bodies ha, non ha i dati degli altri processi — e questo broadcast li rende disponibili.*/
   for (int i = 0; i < iterations; i++) {
    
    // Reset accumuli forze
    memset(sum_dx, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_dy, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_vx, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_vy, 0, bodiesPerProcess[rank] * sizeof(double));

    // Calcolo forze per i corpi locali verso la propria porzione (già la conosco)
  bodyForce(localBodies, localBodies,
          bodiesPerProcess[rank], bodiesPerProcess[rank],
          sum_dx, sum_dy, sum_vx, sum_vy);

// Preallocazione buffer
int maxCount = 0;
for (int p = 0; p < numberOfTasks; p++)
    if (bodiesPerProcess[p] > maxCount)
        maxCount = bodiesPerProcess[p];

Body *sendBuffer = malloc(maxCount * sizeof(Body));
Body *recvBuffer = malloc(maxCount * sizeof(Body));

// Copio i miei corpi nel buffer da inviare
memcpy(sendBuffer, localBodies, bodiesPerProcess[rank] * sizeof(Body));
int sendCount = bodiesPerProcess[rank];
int senderRank = rank;

for (int step = 1; step < numberOfTasks; step++) {
    int next = (rank + 1) % numberOfTasks;
    int prev = (rank + numberOfTasks - 1) % numberOfTasks;

    int recvCount = bodiesPerProcess[(senderRank + numberOfTasks - 1) % numberOfTasks];

    MPI_Sendrecv(
        sendBuffer, sendCount, MPI_BODY, next, 0,
        recvBuffer, recvCount, MPI_BODY, prev, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Calcolo la forza con i corpi ricevuti
    bodyForce(localBodies, recvBuffer,
              bodiesPerProcess[rank], recvCount,
              sum_dx, sum_dy, sum_vx, sum_vy);

    // Preparo per il prossimo step
    Body *tmp = sendBuffer; sendBuffer = recvBuffer; recvBuffer = tmp;
    sendCount = recvCount;
    senderRank = (senderRank + numberOfTasks - 1) % numberOfTasks;
}

free(sendBuffer);
free(recvBuffer);   
    // Ora aggiorno la posizione dei corpi locali
    updatePositions(localBodies, bodiesPerProcess[rank], sum_dx, sum_dy, sum_vx, sum_vy);
   }
        free(sum_dx);
        free(sum_dy);

    // Raccolta dei risultati nel master
    MPI_Gatherv(
        localBodies, bodiesPerProcess[rank], MPI_BODY,
        bodies, bodiesPerProcess, displs, MPI_BODY,
        MASTER, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double finishTime = MPI_Wtime();
    double executionTime = finishTime - startTime;

    if (rank == MASTER) {
        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations,
            executionTime, 1);
    }

    free(bodiesPerProcess);
    free(displs);
    free(localBodies);
    if (rank == MASTER) free(bodies);
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();

    return 0;
}

/*______________________________________________________________________________________________________________________________________________
  ----------FINE MAIN--------------------FINE MAIN------------------FINE MAIN---------------------FINE MAIN-----------------FINE MAIN ----------*/

  void randomizeBodies(Body *bodies, int numberOfBodies) {
    double space = 2e7 * sqrt(numberOfBodies);  // area di distribuzione

    // Per risultati diversi a ogni esecuzione
    srand(42); 

    for (int i = 0; i < numberOfBodies; i++) {
        double scale;

        // Massa positiva realistica
        scale = rand() / (double) RAND_MAX;
        bodies[i].m = 1e20  + scale * (2e20 - 1e20 );

        // Coordinate x, y da -space a +space
        scale = rand() / (double) RAND_MAX;
        bodies[i].x = -space + scale * (2 * space);

        scale = rand() / (double) RAND_MAX;
        bodies[i].y = -space + scale * (2 * space);

        // Velocità vx e vy tra [-50, -5] oppure [5, 50]
        scale = rand() / (double) RAND_MAX;

        if (rand() % 2 == 0) {
        // intervallo negativo
        bodies[i].vx = -50 + scale * (45);  // da -50 a -5
        } else {
        // intervallo positivo
        bodies[i].vx = 5 + scale * (45);    // da 5 a 50
        }
        
        // Velocità vx e vy tra [-50, -5] oppure [5, 50]
        scale = rand() / (double) RAND_MAX;
        if (rand() % 2 == 0) {
        // intervallo negativo
        bodies[i].vy = -50 + scale * (45);  // da -50 a -5
        } else {
        // intervallo positivo
        bodies[i].vy = 5 + scale * (45); 

     }
}
}

void bodyForce(Body *localBodies1, Body *bodies2, int localBodies1Length,int bodies2Length, double *sumDX, double *sumDY,double *sumVX, double *sumVY){
	for(int i = 0; i < localBodies1Length; i++){						                //localBodies[3]={D,E,F}
		for(int j = 0; j<bodies2Length; j++){							                //bodies[15]={A,B,C,D,E,F,G,H,I,L,M,N,O}
			if(localBodies1[i].x!=bodies2[j].x || localBodies1[i].y!=bodies2[j].y ){	//D!=A
				computeForce(localBodies1[i], bodies2[j], &sumDX[i], &sumDY[i], &sumVX[i], &sumVY[i]);
			}
		}
	}
}

void computeForce(Body bodyA, Body bodyB, double *sum_dx, double *sum_dy, double *sum_vx, double *sum_vy){
	double sx = bodyB.x - bodyA.x;
    double sy = bodyB.y - bodyA.y;
    double rAB = sqrt((sx * sx) + (sy * sy)+ SOFTENING);
    //Calcolo dell'accellerazione 
    double acc =  (bodyB.m)*G/(rAB*rAB);   		
    //calcolo del modulo del vettore spostamento
    double d =	sqrt(bodyA.vx *bodyA.vx +bodyA.vy*bodyA.vy)*DT + 0.5* acc * DT * DT; 
  
    double speed = sqrt(bodyA.vx * bodyA.vx + bodyA.vy * bodyA.vy);
    //calcolo delle componenti (x,y) del vettore spostamento
    *sum_dx += (d * sx) / rAB; 
    *sum_dy += (d * sy) / rAB;
    *sum_vx += (sx / rAB) * (speed + acc * DT);
    *sum_vy += (sy / rAB) * (speed + acc * DT);}

void updatePositions(Body *localBodies, int localBodiesLength, double *sum_dx, double *sum_dy, double *sum_vx, double *sum_vy){
	for (int i =0; i < localBodiesLength; i++ ){
		localBodies[i].x += sum_dx[i];
		localBodies[i].y += sum_dy[i];
        localBodies[i].vx +=sum_vx[i];
        localBodies[i].vy +=sum_vy[i];
	}
}

void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks,int *bodiesPerProcess, int *displs) {
    int rest = numberOfBodies % numberOfTasks;
    int bodiesDifference = numberOfBodies / numberOfTasks;
    int startPosition = 0;

    // It's based on the fact that the rest is always less than the divisor
    for (int process = MASTER; process < numberOfTasks; process++) {
        if (rest > 0) {  //se il resto è maggiore di 0
            bodiesPerProcess[process] = bodiesDifference + 1; // ad ogni processo 
            rest--; //viene assegnato un body in più fino al temrinare del resto
        } else {
            bodiesPerProcess[process] = bodiesDifference;
        }

        //                             0                          3                         6                 8
        //dato un array di bodies = [[42,51], [36,47], [34,5], [62,20], [14,94], [78,9], [103,6], [21,13], [68,87]] 
        //displs è un array in cui per ogni porcesso viene memorizzato l'indice iniziale del set di body a lui assegnati
        displs[process] = startPosition; //displs[0] = {0}, displs[1]={0,2}, displs[1]={0,3,6}, displs[1]={0,3,6}
        startPosition += bodiesPerProcess[process]; //0 + 3 = 3,  3 + 3 = 6,  6 + 2 = 8 
    }
}

void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, 
                    double executionTime, int isExecutionTimeRequired) {
    // If execution time is required then it's the end of computation
        printBodies(bodies, numberOfBodies, numberOfTasks, iterations, isExecutionTimeRequired);

    if (isExecutionTimeRequired == 1) {
        printf("BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n", 
            numberOfTasks, numberOfBodies, iterations, executionTime);

        FILE *file = fopen("./nBodyExecutionTime.txt", "a");
        fprintf(file,"BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n\n", 
            numberOfTasks, numberOfBodies, iterations, executionTime);
    }
}

void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd) {
    FILE *file = fopen("./bodies.txt", "a");
    if (!file) {
        perror("Error opening file bodies.txt");
        return;
    }

    if (isEnd == 1) {
        fprintf(file, "BLOCKING: Bodies at the end with %d processors and %d iterations:\n", 
            numberOfTasks, iterations);
    } else {
        fprintf(file, "BLOCKING: Bodies at the beginning with %d processors and %d iterations:\n", 
            numberOfTasks, iterations);
    }

    for (int body = 0; body < numberOfBodies; body++) {
        fprintf(file, "Body[%d][%lf, %lf, %lf, %lf, %lf]\n", body,
            bodies[body].m, bodies[body].x, bodies[body].y, 
            bodies[body].vx, bodies[body].vy);
    }

    fprintf(file, "\n");
    fclose(file);
}

void printHowToUse() {
    printf("To correctly launch nBody run: mpirun -np <Processors> nBody <Bodies> <Iterations> [-R|-G]\n");
}

void readBodies( Body *bodies, int numberOfBodies) {
    FILE *fp = fopen("data.txt", "r");
    if (!fp) {
        perror("Errore apertura file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numberOfBodies; i++) {
        if (fscanf(fp, "Body[%*d][%lf, %lf, %lf, %lf, %lf]\n",
                   &bodies[i].m, &bodies[i].x,
                   &bodies[i].y, &bodies[i].vx, &bodies[i].vy) != 5) {
            fprintf(stderr, "Errore formato alla riga %d\n", i + 1);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fp);
}









