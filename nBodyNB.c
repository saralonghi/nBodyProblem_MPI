#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>


#define DT 0.01
#define SOFTENING 1e-9 
#define MASTER 0

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

    Body *localBodies = (Body*) malloc(bodiesPerProcess[rank] * sizeof(Body));
     

    MPI_Barrier(MPI_COMM_WORLD); // Sincronizza tutti i clock dei processi
    double startTime = MPI_Wtime(); 

    // Scatterv: master invia, ogni processo riceve in buffer locale
    MPI_Scatterv(
        bodies, bodiesPerProcess, displs, MPI_BODY,
        localBodies, bodiesPerProcess[rank], MPI_BODY,
        MASTER, MPI_COMM_WORLD
    );


    double *sum_dx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_dy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   
	double *sum_vx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_vy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   


// numero massimo di corpi da un processo
int maxCount = 0;
for (int p = 0; p < numberOfTasks; p++)
    if (bodiesPerProcess[p] > maxCount)
        maxCount = bodiesPerProcess[p];
int next = (rank + 1) % numberOfTasks;
int prev = (rank - 1 + numberOfTasks) % numberOfTasks;

Body *recvBuffer[2];
recvBuffer[0] = malloc(maxCount * sizeof(Body));
recvBuffer[1] = malloc(maxCount * sizeof(Body));
Body *sendBuffer = malloc(maxCount * sizeof(Body));

for (int i = 0; i < iterations; i++) {
    // Reset accumuli forze
    memset(sum_dx, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_dy, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_vx, 0, bodiesPerProcess[rank] * sizeof(double));
    memset(sum_vy, 0, bodiesPerProcess[rank] * sizeof(double));


    // buffer massimo per sicurezza
    int maxCount = 0;
    for (int p = 0; p < numberOfTasks; p++)
        if (bodiesPerProcess[p] > maxCount)
            maxCount = bodiesPerProcess[p];

    Body *recvBuffer[2];
    recvBuffer[0] = malloc(maxCount * sizeof(Body));
    recvBuffer[1] = malloc(maxCount * sizeof(Body));
    Body *sendBuffer = malloc(bodiesPerProcess[rank] * sizeof(Body));
    memcpy(sendBuffer, localBodies, bodiesPerProcess[rank] * sizeof(Body));

    int next = (rank + 1) % numberOfTasks;
    int prev = (rank + numberOfTasks - 1) % numberOfTasks;

    int senderRank = prev;  // da chi ricevo la prima volta
    int current = 0;

    MPI_Request sendReq, recvReq;

    // --- Prima comunicazione ---
    int recvCount = bodiesPerProcess[senderRank];
    MPI_Isend(sendBuffer, bodiesPerProcess[rank], MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq);
    MPI_Irecv(recvBuffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq);

    // --- Loop principale ---
    for (int step = 1; step < numberOfTasks; step++) {
        // Attendo il blocco ricevuto
        MPI_Wait(&recvReq, MPI_STATUS_IGNORE);

        bodyForce(localBodies, recvBuffer[current], bodiesPerProcess[rank], bodiesPerProcess[senderRank], sum_dx, sum_dy, sum_vx, sum_vy);

        senderRank = (senderRank + numberOfTasks - 1) % numberOfTasks;
        recvCount = bodiesPerProcess[senderRank];

        current = 1 - current;

        MPI_Isend(recvBuffer[1 - current], bodiesPerProcess[(senderRank + 1) % numberOfTasks], MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq);

        MPI_Irecv(recvBuffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq);
    }

    // --- Ultimo blocco ricevuto ---
    MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
    bodyForce(localBodies, recvBuffer[current], bodiesPerProcess[rank], bodiesPerProcess[senderRank], sum_dx, sum_dy, sum_vx, sum_vy);

    // Aggiorno posizioni
    updatePositions(localBodies, bodiesPerProcess[rank], sum_dx, sum_dy, sum_vx, sum_vy);

    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);

    free(sendBuffer);
    free(recvBuffer[0]);
    free(recvBuffer[1]);
}
    free(sum_dx);
    free(sum_dy);
    free(sum_vx);
    free(sum_vy);

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
    *sum_vy += (sy / rAB) * (speed + acc * DT);
}

void updatePositions(Body *localBodies, int localBodiesLength, double *sum_dx, double *sum_dy, double *sum_vx, double *sum_vy){
	for (int i =0; i < localBodiesLength; i++ ){
		localBodies[i].x += sum_dx[i];
		localBodies[i].y += sum_dy[i];
        localBodies[i].vx += sum_vx[i];
        localBodies[i].vy += sum_vy[i];
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
        printf("NON BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n", 
            numberOfTasks, numberOfBodies, iterations, executionTime);

        FILE *file = fopen("./nBodyExecutionTime.txt", "a");
        fprintf(file,"NON BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n\n", 
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
        fprintf(file, "NON BLOCKING: Bodies at the end with %d processors and %d iterations:\n", 
            numberOfTasks, iterations);
    } else {
        fprintf(file, "NON BLOCKING: Bodies at the beginning with %d processors and %d iterations:\n", 
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
    printf("To correctly launch nBody run: mpirun -np <Processors> nBodyNB <Bodies> <Iterations> [-R|-G]\n");
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


