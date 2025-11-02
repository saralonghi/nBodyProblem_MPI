#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>


#define DT 0.01
#define MASTER 0

#define EXPECTED_ARGUMENT 4
#define ARG1_BODIES 1
#define ARG2_ITERATIONS 2
#define ARG3_READ 3

#define CORRECTLY_INVOKED 1
#define NOT_CORRECTLY_INVOKED 0

typedef struct { 
    double m;
    double x;
    double y;
    double vx; 
    double vy; 
} Body;

const double G = 6.67430e-11;  

void generateRandomNumber(Body *bodies, int numberOfBodies);
void computeDeltaState(Body *localBodies1, Body *bodies2, int localBodies1Length,int bodies2Length, double *sumDX, double *sumDY, double *sumVX, double *sumVY);
void twoBodiesForce(Body BodyA, Body BodyB, double *sum_dx, double *sum_dy,double *sum_dvx, double *sum_dvy);
void updateBodiesState(Body *localBodies, int localBodiesLength, double *sum_dx, double *sum_dy, double *sum_dvx, double *sum_dvy);
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfProc, int *bodiesPerProcess, int *displs);
void printExecutionTime(Body *bodies, int numberOfBodies, int numberOfProc, int iterations, double executionTime);
void printBodies(Body *bodies, int numberOfBodies, int numberOfProc, int iterations, int isEnd);
void printCmd();
void readBodies( Body *bodies, int numberOfBodies);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int numberOfProc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProc);
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
            printCmd();
        }
        MPI_Finalize();
        return 0;
    }

    int numberOfBodies = atoi(argv[ARG1_BODIES]);
    int iterations = atoi(argv[ARG2_ITERATIONS]);
    srand(53);

    int bytes = numberOfBodies * sizeof(Body);
    Body *bodies = NULL;

    if (rank == MASTER) {
        bodies = (Body *) malloc(bytes);  

        if(reading == 0){
            generateRandomNumber(bodies, numberOfBodies);  
        }else{
            readBodies(bodies, numberOfBodies);
        }
        printBodies(bodies, numberOfBodies, numberOfProc, iterations, 0);
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
 
    int *bodiesPerProcess = (int*) malloc(numberOfProc * sizeof(int)); 
    int *displs = (int*) malloc(numberOfProc * sizeof(int)); 
      
    buildBodiesPerProcessAndDispls(numberOfBodies, numberOfProc, bodiesPerProcess, displs);

   // MPI_Bcast(bodiesPerProcess, numberOfProc, MPI_INT, 0, MPI_COMM_WORLD);
   // MPI_Bcast(displs, numberOfProc, MPI_INT, 0, MPI_COMM_WORLD);
    Body *localBodies = (Body*) malloc(bodiesPerProcess[rank] * sizeof(Body));
    
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime(); 

    MPI_Scatterv(bodies, bodiesPerProcess, displs, MPI_BODY, localBodies, bodiesPerProcess[rank], MPI_BODY, MASTER, MPI_COMM_WORLD );

    double *sum_dx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_dy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   
	double *sum_dvx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_dvy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   

    int max = bodiesPerProcess[MASTER];
    for (int i = 0; i < iterations; i++) {
        memset(sum_dx, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dy, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dvx, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dvy, 0, bodiesPerProcess[rank] * sizeof(double));
        int max = bodiesPerProcess[0];
        
        Body *buffer[2];
        buffer[0] = malloc(max * sizeof(Body));
        buffer[1] = malloc(max * sizeof(Body));
        int current = 0;
                
        memcpy(buffer[1], localBodies, bodiesPerProcess[rank] * sizeof(Body));
        
        int next = (rank + 1) % numberOfProc;
        int prev = (rank - 1 + numberOfProc) % numberOfProc;

        int senderRank = prev;  
        int recvCount = bodiesPerProcess[senderRank];
        
        int sendCount = bodiesPerProcess[rank]; 

        MPI_Request sendReq, recvReq;

        MPI_Isend(buffer[1], sendCount, MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq); 
        computeDeltaState (localBodies, localBodies, bodiesPerProcess[rank], bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);
        MPI_Irecv(buffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq); // buffer[0]

        for (int step = 1; step < numberOfProc; step++) {
            
            MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
            computeDeltaState (localBodies, buffer[current], bodiesPerProcess[rank], bodiesPerProcess[senderRank], sum_dx, sum_dy, sum_dvx, sum_dvy);// buffer[0]
             
            senderRank = (senderRank + numberOfProc - 1) % numberOfProc;
            recvCount = bodiesPerProcess[senderRank];
            current = 1 - current;// current = 1 
            
            MPI_Isend(buffer[1 - current], bodiesPerProcess[(senderRank + 1) % numberOfProc], MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq);// buffer[0]
            MPI_Irecv(buffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq);// buffer[1]
        }

       
        updateBodiesState(localBodies, bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);

        MPI_Wait(&sendReq, MPI_STATUS_IGNORE);

        free(buffer[0]);
        free(buffer[1]);
    }
    free(sum_dx);
    free(sum_dy);
    free(sum_dvx);
    free(sum_dvy);

    MPI_Gatherv(localBodies, bodiesPerProcess[rank], MPI_BODY, bodies, bodiesPerProcess, displs, MPI_BODY, MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double finishTime = MPI_Wtime();
    double executionTime = finishTime - startTime;

    if (rank == MASTER) {
        
    printBodies(bodies, numberOfBodies, numberOfProc, iterations, 1);
        printExecutionTime(bodies, numberOfBodies, numberOfProc, iterations, executionTime);
    }

    free(bodiesPerProcess);
    free(displs);
    free(localBodies);
    if (rank == MASTER) free(bodies);
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();

    return 0;
}


void generateRandomNumber(Body *bodies, int numberOfBodies) {
    double space = 2e7 * sqrt(numberOfBodies);
    srand(79); 
    for (int i = 0; i < numberOfBodies; i++) {
        double scale;
        scale = rand() / (double) RAND_MAX;
        bodies[i].m = 1e20  + scale * (2e20 - 1e20 );
        scale = rand() / (double) RAND_MAX;
        bodies[i].x = -space + scale * (2 * space);
        scale = rand() / (double) RAND_MAX;
        bodies[i].y = -space + scale * (2 * space);
        scale = rand() / (double) RAND_MAX;

        if (rand() % 2 == 0) {
            bodies[i].vx = -50 + scale * (45);  
        } else {
            bodies[i].vx = 5 + scale * (45);   
        }
        
        scale = rand() / (double) RAND_MAX;
        if (rand() % 2 == 0) {
            bodies[i].vy = -50 + scale * (45);  
        } else {
            bodies[i].vy = 5 + scale * (45); 
        }
    }
}

void computeDeltaState(Body *localBodies1, Body *bodies2, int localBodies1Length,int bodies2Length, double *sumDX, double *sumDY,double *sumVX, double *sumVY){
	for(int i = 0; i < localBodies1Length; i++){						               
		for(int j = 0; j<bodies2Length; j++){							                
			if(localBodies1[i].x!=bodies2[j].x || localBodies1[i].y!=bodies2[j].y ){
				twoBodiesForce(localBodies1[i], bodies2[j], &sumDX[i], &sumDY[i], &sumVX[i], &sumVY[i]);
			}
		}
	}
}

void twoBodiesForce(Body bodyA, Body bodyB, double *sum_dx, double *sum_dy, double *sum_dvx, double *sum_dvy){
    double sx = bodyB.x - bodyA.x;
    double sy = bodyB.y - bodyA.y;
    double rAB = sqrt((sx * sx) + (sy * sy));
    double acc =  (bodyB.m)*G/(rAB*rAB);   
    double v = sqrt(bodyA.vx * bodyA.vx + bodyA.vy * bodyA.vy);
    double d =	v * DT + 0.5* acc * DT * DT; 
    *sum_dx += (d * sx) / rAB; 
    *sum_dy += (d * sy) / rAB;
    *sum_dvx += (sx / rAB) + acc * DT;
    *sum_dvy += (sy / rAB) + acc * DT;
}

void updateBodiesState(Body *localBodies, int localBodiesLength, double *sum_dx, double *sum_dy, double *sum_dvx, double *sum_dvy){
	for (int i =0; i < localBodiesLength; i++ ){
		localBodies[i].x += sum_dx[i];
		localBodies[i].y += sum_dy[i];
        localBodies[i].vx += sum_dvx[i];
        localBodies[i].vy += sum_dvy[i];
	}
}

void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfProc,int *bodiesPerProcess, int *displs) {
    int rest = numberOfBodies % numberOfProc;
    int bodiesDifference = numberOfBodies / numberOfProc;
    int startPosition = 0;

   
    for (int process = MASTER; process < numberOfProc; process++) {
        if (rest > 0) {  
            bodiesPerProcess[process] = bodiesDifference + 1; 
            rest--; 
        } else {
            bodiesPerProcess[process] = bodiesDifference;
        }

   
        displs[process] = startPosition; 
        startPosition += bodiesPerProcess[process];
    }
}

void printExecutionTime(Body *bodies, int numberOfBodies, int numberOfProc, int iterations, double executionTime) {
    printf("NON BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n", 
        numberOfProc, numberOfBodies, iterations, executionTime);

    FILE *file = fopen("./nBodyExecutionTime.txt", "a");
    fprintf(file,"NON BLOCKING: processors %d ,bodies %d, iterations %d ---> time %0.9f seconds\n\n", 
        numberOfProc, numberOfBodies, iterations, executionTime);
}

void printBodies(Body *bodies, int numberOfBodies, int numberOfProc, int iterations, int isEnd) {
    FILE *file = fopen("./bodies.txt", "a");
    if (!file) {
        perror("Error opening file bodies.txt");
        return;
    }

    if (isEnd == 1) {
        fprintf(file, "NON BLOCKING: Bodies at the end with %d processors and %d iterations:\n", 
            numberOfProc, iterations);
    } else {
        fprintf(file, "NON BLOCKING: Bodies at the beginning with %d processors and %d iterations:\n", 
            numberOfProc, iterations);
    }

    for (int body = 0; body < numberOfBodies; body++) {
        fprintf(file, "Body[%d][%lf, %lf, %lf, %lf, %lf]\n", body,
            bodies[body].m, bodies[body].x, bodies[body].y, 
            bodies[body].vx, bodies[body].vy);
    }

    fprintf(file, "\n");
    fclose(file);
}

void printCmd() {
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