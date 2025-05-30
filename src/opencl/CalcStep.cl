
int worldIdx(int i, int j, const int N, const int M){
    i = (i + N) % N;
    j = (j + M) % M;
	return j + i * M;
}

__kernel void calcStep(global int *current, global int *next, int N, int M){
    int gindex = get_global_id(0);
    int threadsi = get_global_size(0); //number of threads

    for(int globali = gindex; globali < N*M; globali += threadsi){
        int i = gindex / M;
        int j = gindex % M;
        //get number of neighbours
        int neighbours = current[worldIdx(i - 1, j - 1, N, M)] + current[worldIdx(i - 1, j, N, M)] + current[worldIdx(i - 1, j + 1, N, M)] +
                        current[worldIdx(i, j - 1, N, M)] + current[worldIdx(i, j + 1, N, M)] +
                        current[worldIdx(i + 1, j - 1, N, M)] + current[worldIdx(i + 1, j, N, M)] + current[worldIdx(i + 1, j + 1, N, M)];

        //set next step 
        next[gindex] = neighbours == 3 || (neighbours == 2 && current[gindex]);        
    }

    

}