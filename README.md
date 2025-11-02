# nBodyProblem_MPI

![img](./doc/copertina.png)

### N-Body Problem – Description and Implemented Solution
The N-body problem involves simulating the motion of N bodies that interact with each other through forces (typically gravitational). This is a classic problem in physics and computational science, with complexity increasing rapidly as the number of bodies grows.

In this project, a simplified version of the problem was implemented by restricting the simulation to two dimensions (x and y) instead of three.

To compute the interactions and update the positions/velocities of the bodies over time, two programs were developed in C, both using the MPI (Message Passing Interface) library to enable distributed computation across multiple processes:

* **Blocking communication version → nBody.c**
  * processes communicate synchronously waiting for the communication to complete before continuing.

* **Non-blocking communication version → nBodyNB.c**
  * processes initiate communication and proceed with computation in parallel, potentially improving performance through better concurrency.

These two versions allow for a comparison of the performance and behavior of blocking vs non-blocking communication approaches in the context of a computationally intensive problem.

___

## Code Descryption
### Blocking program: nBody
In this section will be given explanations about the code.  

When the progam is launched with the command **mpiexec -np 6 nBody 40000 1 -R**, what happens is that six independent process of the same program are run in six different processors. 
When MPI_Init is called, the MPI environment is initialized and all the processes are registered within the global communicator MPI_COMM_WORLD. 
Processes are not aware about the others so it is useful to define, inside each process, the total number of processes sharing the same communicator. 
This is done through the  MPI_Comm_size function, and then with the MPI_Comm_rankeach processor can identify itself within the communicator.
Each process has its owns memory and can interact with other processes just using specific calls.
```C                    
    MPI_Init(&argc, &argv);
    int numberOfProc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```
Previusly a new data struct named Body was defined as follows:
```C
    typedef struct {
        double m; 
        double x; 
        double y; 
        double vx;  
        double vy;  
    } Body; 
```
Then insed the MASTER (process zero), the memory for the bodies is allocated.
It is defined a pointer to a Body type, Body *bodies.
The funcion malloc *(number of bodies **times** the memory occupied by one body), returns a generic pointer void *, thus why it is casted with (Body *). 
If in the command for the execution it was requested to generate the data (-G command) then a function used for creating random double is called, otherwise data will be read. 

```C
     if (rank == MASTER) {
         bodies = (Body *) malloc(numberOfBodies * sizeof(Body));  
    
          if(reading == 0){
              generateRandomNumber(bodies, numberOfBodies);
          }else{
              readBodies(bodies, numberOfBodies);
          }
    }
```
   Now we need to define a MPI data type. An array of integer containing five slots is created each of which is initializate with one. Each one indicates the quantity of the data type needed. 
   Indeed our struct is composed by five doubles. If we had a struct composed by an array of integer, for example int[4] arr, a char and a double, we would have int blocksCount[3] ={4,1,1}
   MPI_Datatype of a struct make use of **padding**, that means that the fields of the struct are not allocated consecutively but ther is an allignment of the memory.
   The alignment depends on the size of the type used. For example, a double, which occupies 8 bytes, needs to be allocated at a multiple of 8.
   The offsets array is an array that contains the offset of each field of the struct. An offset is the distance in byte from the start of the memory. 
   
   ```
        Byte offset:   0  1 2 3 4 5 6 7  8 9 10 11 12 13 14 15
                      [a] [  padding  ]  [       3,4567      ]
   ```
```C
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

    computeDeltaState(localBodies, buffer, bodiesPerProcess[rank], recvCount, sum_dx, sum_dy, sum_dvx, sum_dvy);
```
MPI_Barrier is needed to synchronize the processes, until all the processes within the communicator (in this case MPI_COMM_WORLD) call it the execution of the program is appended. 
Using this function we are sure that alla the processes will start measuring the time from the same point of the execution.

```C     
    MPI_Barrier(MPI_COMM_WORLD); 
    double startTime = MPI_Wtime();
```

The MPI_Scatterv function is used to send to all the processes within the communicator the bodies assigned to each process. 
The parameters used for MPI_Scatterv are: the original buffer containing all the data that have to be spread (bodies),  the array specifying the quantities of data that have to be spread to each process (bodiesPerProcess), 
an array containing the first index of each block of data that we have to send (displs), the type of datawe are sending (MPI_BODY), the receiving buffer (localBodies), the quantity of data that the process have to receive, the type of data that are going to be received the sender (MASTER), and the receiver (MPI_COMM_WORLD).  

```C
    MPI_Scatterv(bodies, bodiesPerProcess, displs, MPI_BODY,localBodies, bodiesPerProcess[rank], MPI_BODY, MASTER, MPI_COMM_WORLD);
``` 

Before starting the for-cycle of the iterations we need to decleare the variables in which the delta space and the delta velcocity will be stored.

```C    
    double *sum_dx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_dy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));   
    double *sum_dvx = (double*) malloc(bodiesPerProcess[rank] * sizeof(double));   
    double *sum_dvy = (double*)malloc(bodiesPerProcess[rank]  * sizeof(double));
```

   For each iteration the delta variables are reset to zero and the maximum number of bodies assigned to a process is computed. 
   The latter step is done in order to ensures that the buffer size is large enough to hold the largest data chunk that might be sent or received during communication.
   Since the MASTER process is the first among those to receive one of the remaining bodies, it is guaranteed to have the highest number of bodies

```C
     for (int i = 0; i < iterations; i++) {
    
        memset(sum_dx, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dy, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dvx, 0, bodiesPerProcess[rank] * sizeof(double));
        memset(sum_dvy, 0, bodiesPerProcess[rank] * sizeof(double));

        computeDeltaState(localBodies, localBodies,bodiesPerProcess[rank], bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);

        int max = bodiesPerProcess[0];

        Body *buffer[2];
        buffer[0] = malloc(max * sizeof(Body));
        buffer[1] = malloc(max * sizeof(Body));
        int current = 0;
```

The local bodies are then copied in the sendBuffer. Other parameters of the MPI_Sendrecv function are computed, such as the number of bodies that have to be sent and the rank of the sender.

```C
      memcpy(buffer[1], localBodies, bodiesPerProcess[rank] * sizeof(Body));
      int sendCount = bodiesPerProcess[rank];
      int senderRank = rank;
```

For each other process, except the ongoing one (step = 1), it is calculated the rank next and the preavius node of the ring and the number of bodies that should be received.
Then the MPI_Sendrecv is called and the new state of the body is computed.

```C
        for (int step = 1; step < numberOfProc; step++) {
            int next = (rank + 1) % numberOfProc;
            int prev = (rank + numberOfProc - 1) % numberOfProc;
            int recvCount = bodiesPerProcess[(senderRank + numberOfProc - 1) % numberOfProc];

            MPI_Sendrecv(buffer[(current+1)%2], sendCount, MPI_BODY, next, 0,buffer[current], recvCount, MPI_BODY, prev, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 
            computeDeltaState(localBodies, buffer[current], bodiesPerProcess[rank], recvCount, sum_dx, sum_dy, sum_dvx, sum_dvy);
```

Reassignment and free of variables. 

```C
			sendCount = recvCount;
            senderRank = (senderRank + numberOfProc - 1) % numberOfProc;
            current=1-current;
		}//endfor

		free(buffer[0]);   
        free(buffer[1]);
        updateBodiesState(localBodies, bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);
    }//endfor

    free(sum_dx);
    free(sum_dy);
	free(sum_dvx);
    free(sum_dvy);
```

Finally, the results are collected with the MPI_Gatherv function into the MASTER process. The final timestamp is recorded and varialbles are freed. 

```C
    MPI_Gatherv(localBodies, bodiesPerProcess[rank], MPI_BODY, bodies, bodiesPerProcess, displs, MPI_BODY, MASTER, MPI_COMM_WORLD );

    MPI_Barrier(MPI_COMM_WORLD);
    double finishTime = MPI_Wtime();
    double executionTime = finishTime - startTime;
	
    free(bodiesPerProcess);
    free(displs);
    free(localBodies);
    if (rank == MASTER) free(bodies);
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();

    return 0;
}
```
___
### Blocking program: nBody
Now we are going to analize all the parts that differs from preavious program, nBody.c

Two buffers are needed for receiving bodies.
```C
	for (int i = 0; i < iterations; i++) {
        ....
		int next = (rank + 1) % numberOfProc;
        int prev = (rank - 1 + numberOfProc) % numberOfProc;

        int senderRank = prev;  
        int recvCount = bodiesPerProcess[senderRank];
        int sendCount = bodiesPerProcess[rank]; 

        Body *buffer[2];
        buffer[0] = malloc(max * sizeof(Body));
        buffer[1] = malloc(max * sizeof(Body));
        Body *sendBuffer = malloc(bodiesPerProcess[rank] * sizeof(Body));
        memcpy(sendBuffer, localBodies, bodiesPerProcess[rank] * sizeof(Body));
```
MPI_Isend is called and the local computation of the new states of the bodies is executed meanwhile the sending operation is ongoing.
The MPI_Irecv is called and then the for cycle is started. As soon as we entrer into the cycle we wait for the preavius calls then the state of the local bodies respect to the bodies just received is evaluated. 
Before the end of the cycle the **rank** of the subsequent **sender**, the quantity of **bodies** that the current process is going to receive and the index of the buffer 
```C
        
        MPI_Request sendReq, recvReq;

        MPI_Isend(buffer[1], sendCount, MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq); 
        computeDeltaState (localBodies, localBodies, bodiesPerProcess[rank], bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);
        MPI_Irecv(buffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq); 

         for (int step = 2; step < numberOfProc; step++) {
            
            MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
            computeDeltaState (localBodies, buffer[current], bodiesPerProcess[rank], bodiesPerProcess[senderRank], sum_dx, sum_dy, sum_dvx, sum_dvy);// buffer[0]
             
            senderRank = (senderRank + numberOfProc - 1) % numberOfProc;
            recvCount = bodiesPerProcess[senderRank];
            current = 1 - current;// current = 1 
            
            MPI_Isend(buffer[1 - current], bodiesPerProcess[(senderRank + 1) % numberOfProc], MPI_BODY, next, 0, MPI_COMM_WORLD, &sendReq);// buffer[0]
            MPI_Irecv(buffer[current], recvCount, MPI_BODY, prev, 0, MPI_COMM_WORLD, &recvReq);// buffer[1]
        }

    MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
    computeDeltaState(localBodies, buffer[current], bodiesPerProcess[rank], bodiesPerProcess[senderRank], sum_dx, sum_dy, sum_dvx, sum_dvy);

    updateBodiesState(localBodies, bodiesPerProcess[rank], sum_dx, sum_dy, sum_dvx, sum_dvy);

    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);

    free(buffer[0]);
    free(buffer[1]);
}
```

____
### Some important functions

The number of bodies are divided per number of processes. If there is a rest, the latter is spread among the processes. In the end not all the processes will have the same number of bodies, but the workload is still well balanced. Then it is defined and initializated and array called displs. In the end each process will have a block of bodies and the displs is used in order to understand when the block starts. 
Each item of displs represent the index of the first body (contained in bodies array) of the block.
Combining the displs[myRank] and the bodiesPerProcess[myRank] we know exactly which is our portion of bodies.

```C
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
```

This fuction takes in input two bodies and the delta variables and evalutaes the new state (position and speed) of the first body related to the force exercised by the second body.
What we find out is not the final result, we need to compare our body to all the others to evaluate its new state. 
The variable **acc** is the accelleration,**s** is the speed, **d** is the displacement of the bodies. Then for both space and speed we need to computed the **x** and **y** vector components.

```C
    void  twoBodiesForce(Body bodyA, Body bodyB, double *sum_dx, double *sum_dy, double *sum_dvx, double *sum_dvy){
	    double sx = bodyB.x - bodyA.x;
        double sy = bodyB.y - bodyA.y;
        double rAB = sqrt((sx * sx) + (sy * sy)+ SOFTENING);

        double acc =  (bodyB.m)*G/(rAB*rAB);   		
        double s = sqrt(bodyA.vx * bodyA.vx + bodyA.vy * bodyA.vy)
        double d =	s * DT + 0.5* acc * DT * DT; 
  
        *sum_dx += (d * sx) / rAB; 
        *sum_dy += (d * sy) / rAB;
        *sum_dvx += (sx / rAB) + acc * DT;
        *sum_dvy += (sy / rAB) + acc * DT;
}
```
In the description of the previous function, we mentioned that performing the calculation on just two bodies is not sufficient; it must be carried out for all of them. This is exactly what this function does. For each body in the localBody array, the new state is computed with respect to all the bodies contained in the buffer.

When both arrays of bodies are localBodies, care must be taken to ensure that a body’s state is not calculated with respect to the force exerted by itself. This is because, in such a case, the radius rAB would be zero, and therefore the term G * mB /(rAB*rAB) would tend to infinity.
```C
    void computeDeltaState(Body *localBodies1, Body *bodies2, int localBodies1Length,int bodies2Length, double *sumDX, double *sumDY,double *sumVX, double *sumVY){
	    for(int i = 0; i < localBodies1Length; i++){						               
		    for(int j = 0; j<bodies2Length; j++){							                
			    if(localBodies1[i].x!=bodies2[j].x || localBodies1[i].y!=bodies2[j].y ){	
				     twoBodiesForce(localBodies1[i], bodies2[j], &sumDX[i], &sumDY[i], &sumVX[i], &sumVY[i]);
			    }
		    }
	    }
    }
```




