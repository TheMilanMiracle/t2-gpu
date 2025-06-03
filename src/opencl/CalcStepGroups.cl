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

// groups are of block_size by block_size
#define block_size 8


/** 
    Calculates a step on conway's game of life
    @param currentGlobal global array representing current state of world
    @param next global array representing next state of world
    @param N size of world's x axis
    @param M size of world's y axis
*/
__kernel void calcStep(__global int *currentGlobal, __global int *next, int N, int M){
    
    
    // The cells on a corner of the group write 4 cells, themselfs and
    // the other 3 that touches it (outside of group area)
    // The cells in a border but not a corner write themselfs and 
    // the one that its touching (outside)
    // Any other cell in the interior of the group only writes themselfs.

    // position of thread in the group
    int igroup = get_local_id(0), jgroup = get_local_id(1);
    // size of group
    int nigroupOG = get_local_size(0), njgroupOG = get_local_size(1);

    //position on global world
    int globaliOG = get_global_id(0);
    int globaljOG = get_global_id(1);

    // the local buffer should be (2 + groupDim(0)) x (2 + groupDim(1)) 
    local int currentLocal[(block_size+2) * (block_size+2)];

    //position on local memory (considering padding)
    int iloc = igroup+1, jloc = jgroup+1;

    //adjust group size for incomplete groups
    int offseti = max((globaliOG + (nigroupOG - (igroup + 1))) - N + 1, 0);
    int offsetj = max((globaljOG + (njgroupOG - (jgroup + 1))) - M + 1, 0);
    int nigroup = nigroupOG - offseti, njgroup = njgroupOG - offsetj;

    // size of local memory 
    int niloc = nigroup+2, njloc = njgroup+2;

    //copies itself to local memory
    currentLocal[worldIdx(iloc, jloc, niloc, njloc)] = currentGlobal[worldIdx(globaliOG, globaljOG, N, M)];
    
    //calculates number of extra copies
    int copies =  (igroup == 0 && jgroup == 0) +            //corners, just 1
                (igroup == 0 && jgroup == njgroup-1) +       
                (igroup == nigroup-1 && jgroup == 0) +
                (igroup == nigroup-1 && jgroup == njgroup-1) +
                (igroup == 0) +                         //borders, could be 1 or 2
                (igroup == nigroup-1) +
                (jgroup == 0) +
                (jgroup == njgroup-1);
    
    // traslation on i and j depending on group position
    int borderI = (igroup == 0) * -1 + (igroup == nigroup-1); 
    int borderJ = (jgroup == 0) * -1 + (jgroup == njgroup-1);

    // if copies == 3, will copy all this positions
    // otherwise, its just a border and will only copy the first one.
    int positions[6] = {borderI, borderJ, 
                    borderI, 0,
                    0, borderJ};


    for(int k = 0; k < copies*2; k+=2){
        currentLocal[worldIdx(iloc + positions[k], jloc + positions[k+1], niloc, njloc)] = currentGlobal[worldIdx(globaliOG + positions[k], globaljOG + positions[k+1], N, M)];
    }

    // we wait everyone to make the copies
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now the cell can access its 9 neighbors faster...
    //Since half of them only copied 1 cell... should be faster

    //get number of neighbours
    int neighbours = currentLocal[worldIdx(iloc - 1, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc - 1, jloc, niloc, njloc)] + currentLocal[worldIdx(iloc - 1, jloc + 1, niloc, njloc)] +
                    currentLocal[worldIdx(iloc, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc, jloc + 1, niloc, njloc)] +
                    currentLocal[worldIdx(iloc + 1, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc + 1, jloc, niloc, njloc)] + currentLocal[worldIdx(iloc + 1, jloc + 1, niloc, njloc)];

    //set next step
    next[worldIdx(globaliOG, globaljOG, N, M)] = neighbours == 3 || (neighbours == 2 && currentLocal[worldIdx(iloc, jloc, niloc, njloc)]);
}