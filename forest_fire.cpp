#include <mpi.h> //MPI library for parallel processing
#include <iostream> //I/O operations
#include <fstream> //file handling
#include <vector> //dynamic arrays, fire cells
#include <cstdlib> //standard operations
#include <ctime> //time functions
#include <string> //string manipulation
#include <sstream> //string stream operations
#include <algorithm> //swapping

using namespace std;

//cnstants for cell states
const int EMPTY   = 0; //empty cell
const int TREE    = 1; //living tree
const int BURNING = 2; //burning tree
const int BURNT   = 3; //burnt tree(ash)

//generate a random 2D grid of size NxN with probability p of having a tree in each cell
void generateGrid(vector<vector<int>> &grid, int N, double p) {
    grid.resize(N, vector<int>(N, EMPTY)); //resize the grid to N rows, each row having N columns; initialise all cells to EMPTY
    //fill the grid randomly based on probability p
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double r = static_cast<double>(rand()) / RAND_MAX; //random fraction in [0,1]
            grid[i][j] = (r < p) ? TREE : EMPTY; //TREE if r < p, otherwise EMPTY
        }
    }
    //top row is ignited: if a cell in row 0 is a TREE, set it to BURNING
    for(int j = 0; j < N; j++){
        if(grid[0][j] == TREE){
            grid[0][j] = BURNING;
        }
    }
}

//read the given grid in .txt format; the first integer is N, followed by N*N states (0,1,2,3)
void readGrid(const string &filename, vector<vector<int>> &grid, int &N) {
    ifstream infile(filename); //open file
    if(!infile){
        cerr << "Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); //abort if file read fails
    }
    infile >> N; //first intiger in the file is the grid size
    grid.resize(N, vector<int>(N, EMPTY)); //resize to N×N, default to empty
    //read N×N cell values from the file
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            infile >> grid[i][j];
        }
    }
    //ignite top row: if a cell in row 0 is TREE, set it to BURNING
    for(int j = 0; j < N; j++){
        if(grid[0][j] == TREE) {
            grid[0][j] = BURNING;
        }
    }
    infile.close(); //close file
}

//perform local simulation of the forest fire, updating local_current until no fire remains
int simulateLocal(int local_rows, int N, // local_rows is how many "real" rows the process has (not counting the 2 ghost rows)
                  vector<vector<int>> &local_current, // local_current has local_rows+2 rows each of width N
                  vector<vector<int>> &local_next, //used to build the state for the next time-step
                  int rank, int size, bool &local_bottom_reached) //defines the number of ranks and wether the fire reaches the bottom
{
    int steps = 0; //count how many steps the fire simulation takes
    bool fire_active = true; //ccheck if fire still remains anywhere in the system

    while(fire_active){
        MPI_Status status;

        //exchange ghost rows with neighbors
        //send row 1 up to rank-1 (receiving into row 0)
        if(rank != 0){
            MPI_Sendrecv(&local_current[1][0], //send
                         N, MPI_INT,
                         rank - 1, 0,
                         &local_current[0][0], //receive
                         N, MPI_INT,
                         rank - 1, 0,
                         MPI_COMM_WORLD, &status);
        }
        //send row local_rows down to rank+1 (receiving into row local_rows+1)
        if(rank != size - 1){
            MPI_Sendrecv(&local_current[local_rows][0], //send (last row)
                         N, MPI_INT,
                         rank + 1, 0,
                         &local_current[local_rows + 1][0], //receive (bottom ghost row)
                         N, MPI_INT,
                         rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        }

        bool local_fire = false;
        //update the "real" rows [1..local_rows], ignoring the ghost rows [0] and [local_rows+1]
        for(int i = 1; i <= local_rows; i++){
            for(int j = 0; j < N; j++){
                int cell = local_current[i][j];
                if(cell == BURNING){
                    //burning cell becomes BURNT, turn to ash
                    local_next[i][j] = BURNT;
                }
                else if(cell == TREE){
                    //check Von Neumann neighbors for fire; tree catches fire if any neighbor is BURNING
                    bool neighborBurning = false;
                    // up
                    if(local_current[i - 1][j] == BURNING) neighborBurning = true;
                    // down
                    if(local_current[i + 1][j] == BURNING) neighborBurning = true;
                    // left
                    if(j > 0 && local_current[i][j - 1] == BURNING) neighborBurning = true;
                    // right
                    if(j < N - 1 && local_current[i][j + 1] == BURNING) neighborBurning = true;
                    local_next[i][j] = (neighborBurning ? BURNING : TREE);
                }
                else{
                    local_next[i][j] = cell; //EMPTY or BURNT remain unchanged
                }
                //check if any cell is still on fire after the update
                if(local_next[i][j] == BURNING) {
                    local_fire = true;
                }
            }
        }

        //swap local_current and local_next for the next iteration
        swap(local_current, local_next);

        //check if fire remains
        int lf = local_fire ? 1 : 0; //local fire
        int gf = 0; //global fire
        MPI_Allreduce(&lf, &gf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
       //if the sum of local_fire is zero, no fire remains
        if(gf == 0) {
            fire_active = false;
        }
        steps++; //move to the next step
    }

    //check if the fire reached the bottom row (only last rank checks)
    local_bottom_reached = false;
    if(rank == size - 1 && local_rows > 0){
        //check the real bottom row in local_current
        for(int j = 0; j < N; j++){
            int st = local_current[local_rows][j];
            if(st == BURNT || st == BURNING){
                local_bottom_reached = true;
                break;
            }
        }
    }
    return steps;
}

//entry point for the overall simulation
int main(int argc, char** argv){
    MPI_Init(&argc, &argv); //initialising MPI environment
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //current process number
    MPI_Comm_size(MPI_COMM_WORLD, &size); //total number of processes

    int N = 100, M = 50; //given parameters for the grid size and number of runs; can be overrriden by command-line arguments
    double p = 0.6; //probability of a cell having a tree
    string inputFile; //optional input file path

    //parsing command-line arguments
    for(int i = 1; i < argc; i++){
        string arg = argv[i];
        //parse grid size N
        if(arg == "-n" && i + 1 < argc) {
            N = atoi(argv[++i]);
        }
        //parse starting probability p
        else if(arg == "-p" && i + 1 < argc) {
            p = atof(argv[++i]);
        }
        //parse number of runs M
        else if(arg == "-m" && i + 1 < argc) {
            M = atoi(argv[++i]);
        }
        //parse input file
        else if(arg == "-f" && i + 1 < argc) {
            inputFile = argv[++i]; //store file name
        }
    }

    //store final results from each run
    vector<int> all_steps(M, 0), all_bottom(M, 0); //steps for each run & whether bottom was reached (0 or 1)
    vector<double> all_times(M, 0.0); //time for each run

    //open a CSV file to store the results
    ofstream csvFile;
    if(rank == 0){
        //create a filename based on the parameters
        ostringstream fname;
        fname << "results_N_" << N << "_p_" << p << "_M_" << M << ".csv";
        csvFile.open(fname.str());
        if(!csvFile.is_open()){
            cerr << "Error: could not open CSV file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csvFile << "Run,Steps,ReachedBottom,SimulationTime\n"; //csv header
    }

    //variables accumulating data for computing averages
    double totSteps = 0.0, totTime = 0.0;
    int totBottom = 0;

    //main simulation loop over M runs
    for(int run = 0; run < M; run++){
        //random number generator, different for each run
        if(rank == 0){
            srand(static_cast<unsigned>(time(NULL)) + run);
        }

        //read a grid from file or generates a random one
        vector<vector<int>> grid;
        if(rank == 0){
            if(!inputFile.empty()){
                readGrid(inputFile, grid, N); //read the grid from input file
            }
            else {
                generateGrid(grid, N, p); //generate a random grid
            }
        }

        //broadcast N in case it changed with readGrid
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //distribute rows across MPI ranks
        int rows_per_proc = N / size; 
        int remainder = N % size; //leftover rows
        int local_rows = rows_per_proc + ((rank < remainder) ? 1 : 0);

        //each rank receives local_rows*N data elements
        vector<int> local_data(local_rows * N);

        //scatter the entire grid from rank 0 to all ranks
        if(rank == 0){
            //flatten the 2D grid to a 1D vector
            vector<int> full_data;
            full_data.reserve(N * N);
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    full_data.push_back(grid[i][j]);
                }
            }

            //arrays for MPI_Scatterv: sendcounts and displacements
            vector<int> sendcounts(size), displs(size);
            int offset = 0;
            for(int i = 0; i < size; i++){
                int rp = rows_per_proc + ((i < remainder) ? 1 : 0);
                sendcounts[i] = rp * N;
                displs[i] = offset;
                offset += rp * N;
            }
            //scatter data to all ranks according to sendcounts/displacements
            MPI_Scatterv(full_data.data(), sendcounts.data(), displs.data(), MPI_INT,
                         local_data.data(), local_rows * N, MPI_INT,
                         0, MPI_COMM_WORLD);
        }
        else {
            //ranks other than 0 receive their portion
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                         local_data.data(), local_rows * N, MPI_INT,
                         0, MPI_COMM_WORLD);
        }

        //create local grids with 2 extra ghost rows: top(0) & bottom(local_rows+1)
        vector<vector<int>> local_current(local_rows + 2, vector<int>(N, EMPTY));
        vector<vector<int>> local_next   (local_rows + 2, vector<int>(N, EMPTY));

        //fill local_current[1..local_rows] from local_data (excluding ghost rows)
        for(int i = 0; i < local_rows; i++){
            for(int j = 0; j < N; j++){
                local_current[i + 1][j] = local_data[i * N + j];
            }
        }

        //if this is the top rank, row 0 is not used for real data -> set to EMPTY
        if(rank == 0) {
            for(int j = 0; j < N; j++){
                local_current[0][j] = EMPTY;
            }
        }
        //if this is the bottom rank, row local_rows+1 is a bottom ghost row -> set to EMPTY
        if(rank == size - 1) {
            for(int j = 0; j < N; j++){
                local_current[local_rows + 1][j] = EMPTY;
            }
        }

        //synchronize all ranks, start timing the local simulation
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        bool lb = false;  // local_bottom_reached
        int steps = simulateLocal(local_rows, N, local_current, local_next, rank, size, lb);

        double end = MPI_Wtime();
        double stime = end - start;  // local simulation time on this rank

        //combine results
        //1. steps
        int global_steps = 0; //the maximum local steps iterated
        use MPI_MAX because each rank sees different step counts,
        //MPI_MAX because each rank sees different step counts
        MPI_Reduce(&steps, &global_steps, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        //2. did the fire reach bottom?
        int bf = lb ? 1 : 0;
        int gbSum = 0;
        MPI_Reduce(&bf, &gbSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        int global_bottom = (gbSum > 0) ? 1 : 0;

        //3. gather timings
        double global_time = 0.0;
        MPI_Reduce(&stime, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        //on rank 0, record the results and print them
        if(rank == 0){
            all_steps[run]   = global_steps;
            all_bottom[run]  = global_bottom;
            all_times[run]   = global_time;

            totSteps  += global_steps;
            totTime   += global_time;
            totBottom += global_bottom;

            cout << "Run " << (run + 1)
                 << ": Steps = " << global_steps
                 << ", Reached bottom: " << (global_bottom ? "Yes" : "No")
                 << ", Time = " << global_time << " s" << endl;

            if(csvFile.is_open()){
                csvFile << (run + 1) << ","
                        << global_steps << ","
                        << global_bottom << ","
                        << global_time << "\n";
            }
        }
    }

    //final summary on rank 0
    if(rank == 0){
        double avg_steps = (M > 0) ? (totSteps / M) : 0.0;
        double avg_time  = (M > 0) ? (totTime  / M) : 0.0;
        double frac      = (M > 0) ? (100.0 * totBottom / M) : 0.0;

        cout << "\n=== Averages over " << M << " runs ===" << endl;
        cout << "Average steps: " << avg_steps << endl;
        cout << "Fire reached bottom in " << totBottom << " / " << M
             << " runs (" << frac << "%)" << endl;
        cout << "Average simulation time: " << avg_time << " s" << endl;

        //write a final summary line to the CSV and close it
        if(csvFile.is_open()){
            csvFile << "Average," << avg_steps
                    << ",(fraction_bottom=" << frac << "%),"
                    << avg_time << "\n";
            csvFile.close();
        }
    }

    //finalize the MPI environment
    MPI_Finalize();
    return 0;
}