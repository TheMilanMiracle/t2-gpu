/** 
    Calculates 1d coordinates from 2d coordinates
    @param i position on x axis
    @param j position on y axis
    @param N size of x axis
    @param M size of y axis
*/
int worldIdx(int i, int j, const int N, const int M){
    i = (i + N) % N;
    j = (j + M) % M;
	return j + i * M;
}

/** 
    Calculates a step on conway's game of life
    @param currentGlobal global array representing current state of world
    @param next global array representing next state of world
    @param N size of world's x axis
    @param M size of world's y axis
*/
__kernel void calcStep(global int *current, global int *next, int N, int M){
    
    // global position on each dimension
    int globali = get_global_id(0), globalj = get_global_id(1);
   
    //get number of neighbours
    int neighbours = current[worldIdx(globali - 1, globalj - 1, N, M)] + current[worldIdx(globali - 1, globalj, N, M)] + current[worldIdx(globali - 1, globalj + 1, N, M)] +
                    current[worldIdx(globali, globalj - 1, N, M)] + current[worldIdx(globali, globalj + 1, N, M)] +
                    current[worldIdx(globali + 1, globalj - 1, N, M)] + current[worldIdx(globali + 1, globalj, N, M)] + current[worldIdx(globali + 1, globalj + 1, N, M)];

    //set next step 
    next[worldIdx(globali, globalj, N, M)] = neighbours == 3 || (neighbours == 2 && current[worldIdx(globali, globalj, N, M)]);    
}