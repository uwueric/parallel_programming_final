// module load openmpi/4.1.5-gcc
// Usage: mpirun -np <num_nodes> ./p2 <dir> <nthreads> <nreaders> <nmappers> <nnum_reducers>

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

#define MAX_WORD 60
#define MAX_FILES 1024
#define MAX_PATH 256
#define MAX_UNIQUE 200000

using namespace std;

/* RingBuffer implementation from Kayode! */
/* I hate header files so I'm directly putting this here */
/* Adjusted for MPI -> added mutexes and locks to prevent deadlocks */
// https://codereview.stackexchange.com/questions/280956/sender-receiver-threads-using-stdunique-lock-and-stdcondition-variable
// https://en.cppreference.com/w/cpp/thread/unique_lock.html
class RingBuffer {
public:
    RingBuffer() : closed(false) {}

    void rb_push(const pair<string,int>& p) {
        {
            unique_lock<mutex> lock(mutex);
            items.push(p);
        }
        cv.notify_one(); // https://en.cppreference.com/w/cpp/thread/condition_variable/notify_one
    }

    /* We'll block until an item is available or the queue is closed! */
    /* Returns to the mappers' -> { "", -1 } if closed and empty */
    pair<string,int> rb_pop() {
        unique_lock<mutex> lock(mutex);
        cv.wait(lock, [&]{ return closed || !items.empty(); });
        if (items.empty()) return {"", -1}; // closed and empty
        auto out = items.front();
        items.pop();
        return out;
    }

    /* Close the queue and wake all waiting threads! */
    /* We'll use this to put an end to a phase so all waiting threads can move to the next phase */
    void close() {
        {
            unique_lock<mutex> lock(mutex);
            closed = true;
        }
        cv.notify_all();
    }

private:
    queue<pair<string,int>> items; // Items for mapper to 'pop' from reader
    mutex mutex;
    condition_variable cv;
    bool closed;
};

/* output record struct to hold all info for reducers to process! (trackks reducer stats for a specific node) */
/* mapper -> reducer phase use */
struct OutRecord {
    int local_reducer;  // reducer index within the destination node
    string word;        // the word itself will be used as key!
    int count;          // well yes!
};

/* Hash function! */
/* Each reducer has its own subset of keys (words), and we use a hash function to */
/* filter them into "buckets" for the reducer */
/* MAPPER THREAD work! */
unsigned int djb_hash(const string &s, int num_reducers) {
    unsigned long hash = 5381;
    for (unsigned char c : s)
        hash = ((hash << 5) + hash) + c;
    return (unsigned int)(hash % num_reducers);
}

/* Trim function! */
/* Removes puncutation from start/end of word */
/* READER THREAD work! */
string trim_punct(const string &s) {
    if (s.empty()) return s;
    size_t start = 0;
    while (start < s.size() && ispunct((unsigned char)s[start])) start++;
    size_t end = s.size();
    while (end > start && ispunct((unsigned char)s[end-1])) end--;
    return s.substr(start, end - start);
}

/* Splits a line into multiple words! */
/* READER THREAD work! */
static void process_line(const string &line, RingBuffer &rb) {
    istringstream is(line);
    string token;

    while (getline(is, token, ' ')) {
        string w = trim_punct(token);
        if (!w.empty()) rb.rb_push({w,1});
    }

}

/* Reading in all the files! */
/* In the following function calls, everything will get pushed into 'rb' */
/* READER THREAD work! */
void reader(const char *path, RingBuffer &rb) {
    string line;
    ifstream f(path);

    if (!f.is_open()) {
        fprintf(stderr, "reader: invalid file %s\n", path);
        return;
    }

    while (getline(f, line))
        process_line(line, rb);

    f.close();

}

/* Time to slay!! */
int main(int argc, char **argv) {

    /* Initialize MPI!! --------------------------------------------------- */
    MPI_Init(&argc,&argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /* Basic error check of command line arguments  ---------------------------------------------- */
    if(argc < 6){
        if (rank == 0) fprintf(stderr,"Usage: %s <dir> <nthreads> <nreaders> <nmappers> <nreducers>\n", argv[0]);
        
        MPI_Finalize(); 
        return 1;
    }

    /* Process the command line arguments! ----------------------------------------------------- */
    char path[MAX_PATH]; 
    strncpy(path, argv[1], MAX_PATH - 1); 
    path[MAX_PATH - 1] = 0;

    int omp_threads = atoi(argv[2]); // total number of threads
    int num_readers = atoi(argv[3]); // number of reader threads
    int num_mappers = atoi(argv[4]); // number of mapper threads
    int num_reducers = atoi(argv[5]); // number of reducer threads per node
    int total_reducers = num_reducers * size;

    /* Basic debugging! */
    /* just in case something goes wrong w/ inputs */
    if (omp_threads <= 0) omp_threads = 1;
    if (num_readers <= 0) num_readers = 1;
    if (num_mappers <= 0) num_mappers = 1;
    omp_set_num_threads(omp_threads);

    /* Process all the files! ----------------------------------- */
    /* Okay so, the master thread will be responsible for opening all the files and beginning our work process */
    vector<string> local_files;

    if (rank == 0){

        /* Open the directory! */
        DIR *dp = opendir(path);
        if (!dp){ 
            fprintf(stderr,"Cannot open directory %s\n", path); 

            MPI_Finalize(); 
            return 1; 
        }

        /* Go through the list of files within it! */
        /* Hey we did this in 360 before!! */
        struct dirent *ep;
        vector<string> file_list;
        
        while((ep = readdir(dp))){
            if (ep->d_name[0] == '.') continue;
            file_list.push_back(string(path)+"/"+ep->d_name);
        }
        closedir(dp);

        printf("Master found %zu files\n", file_list.size());

        /* Reached here? We have all the files in the master thread! */
        /* Let's distribute these files across all the reader threads */
        int next_file = 0;
        int num_done = 0;
        MPI_Status status;

        /* We'll continuously send the files w/ the master thread */
        /* Send info -> length of file and the actual file */
        /* AND if we finished sending all the files, send a done so the reader threads */
        /* know to stop listening! */
        while(num_done < size - 1){

            /* Master thread listens for requests from reader threads before sending anything! */
            MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int dst = status.MPI_SOURCE;

            /* Because Recv blocks, once we reach here, we know we have a request to process! */
            /* Send the info!! */
            if (next_file < (int)file_list.size()){
                int flen = (int)file_list[next_file].size();

                MPI_Send(&flen, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
                MPI_Send(file_list[next_file].c_str(), flen, MPI_CHAR, dst, 0, MPI_COMM_WORLD);

                next_file++; // Increments with each file sent so we can keep track of when to send done flags
            } 
            /* Reached here? All files are done processing! Send the done flags! */
            else {
                int done_flag = -1; // our "all done message!""

                MPI_Send(&done_flag,1,MPI_INT,dst,0,MPI_COMM_WORLD);

                num_done++; // We have to make sure each thread gets a done flag so we have no blocked threads
            }
        }

        if (next_file < (int)file_list.size())
            local_files.assign(file_list.begin() + next_file, file_list.end());
    } 
    /* Here, all non-master reader threads send requests to the master thread for files! */
    else {

        while(true){

            MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

            /* Now, let's start getting things ready for receiving the file info! */
            int fname_len;
            char buf[MAX_PATH];

            /* Receive the info!! */
            MPI_Recv(&fname_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (fname_len == -1) break;
            MPI_Recv(buf, fname_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            buf[fname_len] = 0;
            local_files.push_back(string(buf));
        }

    }

    /* Ensure that all reader threads have received their appropriate files */
    /* before we move on! */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Now, let's setup our initial queues for reducing and buffers for sending over MPI! -------- */
    RingBuffer work_q; // Reader → Mapper queue (local)
    vector<RingBuffer> reducer_q(num_reducers); // Mapper → Reducer queues (local)
    vector<vector<OutRecord>> send_buffers(size); // Mapper → MPI send buffers (per node)

    /* Timing variables!! */
    double t_total_start = MPI_Wtime(); // Total
    double t_reader_start = 0; // Readers
    double t_reader_end = 0;
    double t_mapper_start = 0; // Mappers
    double t_mapper_end = 0;
    double t_comm_start = 0; // Sending info from mapper -> reducer
    double t_comm_end = 0;
    double t_reducer_start = 0; // Reducers
    double t_reducer_end = 0;

    /* File distribution (READER) time! --------------------------------- */
    /* Each thread will process their own subset of files */
    /* and call the loop of labeled functions 200 lines above! */
    MPI_Barrier(MPI_COMM_WORLD);
    t_reader_start = MPI_Wtime();

    /* Dynamic would probably be more effecient here */
    /* but I'm not sure how it'll know to balance the workload of files */
    /* Setup our parallel region w/ threads = num_readers! */
    #pragma omp parallel for schedule(static) num_threads(num_readers)
    for (int i = 0; i < local_files.size(); i++){
        reader(local_files[i].c_str(), work_q);
    }

    work_q.close(); // Conclude this phase!

    t_reader_end = MPI_Wtime();
    
    MPI_Barrier(MPI_COMM_WORLD);

    /* Hash and locally reduce (MAPPER) time! --------------------------------- */
    t_mapper_start = MPI_Wtime();

    /* Setup our parallel region w/ threads = num_mappers! */
    #pragma omp parallel num_threads(num_mappers)
    {
        unordered_map<string,int> local_map;

        /* Here, each thread will add a word into their local reducer queue */
        /* The goal is to both save all the words into their queue for the reducing phase */
        /* AND doing their own local reducing before sending it to the more 'global' reducer threads */
        while (true) {

            /* Pop a word from the reader work queue! */
            /* It's kinda like a video game LMAO */
            auto item = work_q.rb_pop();

            if (item.second == -1) break; // Once we see EOF, the work for that thread is done!
            if (!item.first.empty()) local_map[item.first] += item.second; /* Found an item? See if we can increment its count! */

        }

        /* Now, it's time to map each word and reducer to a specific thread! */
        /* This is how all the words propagate correctly into one thread rather than being spread out */
        #pragma omp critical
        {
            for (auto &p : local_map){

                unsigned int global_r = djb_hash(p.first, total_reducers);

                int dest_node = global_r / num_reducers;
                int local_r = global_r % num_reducers;

                send_buffers[dest_node].push_back({local_r, p.first, p.second});
            }
        }
    }

    t_mapper_end = MPI_Wtime(); // Conclude this phase!
    MPI_Barrier(MPI_COMM_WORLD);

    /* Send things to the right local reducer (COMMUNICATION) time! --------------------------------- */
    t_comm_start = MPI_Wtime();

    /* We have to start "packing" the information into buffers! */
    /* Once we reached here, all local threads have mapped their subset of words to a reducer */
    /* So thread 1 could have figured out that the word 'apple' in their posession maps to */
    /* a reducer in thread 2. This means that thread 1 now has to send 'apple' and its count over to */
    /* thread 2 so that it can reduce, especially if other threads send 'apple' to it as well! */
    vector<vector<char>> send_payloads(size);

    /* This means that we now have to send all the data that our OutRecord struct holds! */
    /* Word length, the actual word, the local reducer, and the count! */
    for (int i = 0; i < size; i++){ // i -> destination index

        auto &buf = send_payloads[i];
        int nrec = (int)send_buffers[i].size();

        buf.insert(buf.end(), (char*)&nrec, (char*)&nrec + sizeof(int)); // nrec -> number of records for this destination!

        for(auto &r : send_buffers[i]){

            /* Now, for each record, we're sending all the info for the reducer Struct (local reducer, count, the acctual word, and the length of the word)! */
            buf.insert(buf.end(), (char*)&r.local_reducer, (char*)&r.local_reducer + sizeof(int));
            buf.insert(buf.end(), (char*)&r.count, (char*)&r.count + sizeof(int));

            int wlen = (int)r.word.size();
            buf.insert(buf.end(), (char*)&wlen, (char*)&wlen + sizeof(int));

            if (wlen > 0){
                buf.insert(buf.end(), r.word.data(), r.word.data() + wlen);
            }

        }
    }

    /* Reached here? The buffer with our data should have everything it needs now! */
    /* Now we just have to prepare our send/rec buffers (get the right indexes as always) for the MPI functioons! */
    vector<int> send_counts(size);
    vector<int> send_displs(size, 0);
    vector<int> recv_counts(size);
    vector<int> recv_displs(size, 0);
    int total_send = 0;

    for (int i = 0; i < size; i++){ 
        send_counts[i] = (int)send_payloads[i].size(); 
        send_displs[i] = total_send; 
        
        total_send += send_counts[i]; 
    }

    /* Now we'll coalesce the above two into one sendbuffer! */
    vector<char> sendbuf(total_send);
    for (int i = 0; i < size; i++){
        if (send_counts[i] > 0){
            memcpy(sendbuf.data() + send_displs[i], send_payloads[i].data(), send_counts[i]);
        }
    }

    /* and send! */
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,MPI_COMM_WORLD);

    /* Now, we'll have to do the same but for receiving */
    /* Prepare the buffers to store the count, the word length, the word itself, and */
    /* push all of that into the buffer of the local reducer associated with it! */
    int total_recv = 0;

    for (int i = 0; i < size; i++){ 
        recv_displs[i] = total_recv; 
        total_recv += recv_counts[i]; 
    }

    /* Build our receivebuffer! */
    vector<char> recvbuf(total_recv);

    MPI_Alltoallv(sendbuf.data(), send_counts.data(), send_displs.data(), MPI_CHAR, recvbuf.data(), recv_counts.data(), recv_displs.data(),MPI_CHAR,MPI_COMM_WORLD);

    /* Now let's copy all the received info into local variables and */
    /* place them into the buffer of the associated local reducer!! */
    size_t pos = 0;
    
    while(pos < recvbuf.size()){

        /* After every round of information processed, we incrememt the position inside the receive buffer to grab more data */
        /* Each data, with the exception of the word itself, can be fit within the size of an int */
        /* this means that we don't have to do a bunch of send/recvs and can just mathematically figure out the */
        /* position of all the data from the buffer! */
        int nrec = 0; 
        memcpy(&nrec, recvbuf.data() + pos, sizeof(int)); 
        pos += sizeof(int);

        /* Put the actual data in!! */
        for (int i = 0; i < nrec; i++){

            int local_r = 0;
            int cnt = 0;
            int wlen = 0;

            memcpy(&local_r, recvbuf.data() + pos, sizeof(int)); // local reducer
            pos += sizeof(int);
            memcpy(&cnt, recvbuf.data() + pos, sizeof(int)); // count
            pos += sizeof(int);
            memcpy(&wlen, recvbuf.data() + pos, sizeof(int)); // word length
            pos += sizeof(int);

            string word;

            if (wlen > 0){ 
                word.assign(recvbuf.data() + pos, recvbuf.data() + pos + wlen); 
                pos += wlen; 
            }
            
            /* Finally, after all that math... we push it into the respective local reducers' queues! */
            reducer_q[local_r].rb_push({word, cnt});

        }
    }

    /* Let the communication stop and move on! */
    for (int i = 0; i < num_reducers; i++){
        reducer_q[i].close();
    }

    t_comm_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    /* Reduceeeeeeeeeeee everything!! (REDUCER) time! --------------------------------- */
    t_reducer_start = MPI_Wtime();

    vector<unordered_map <string, int>> local_maps(num_reducers);
    int local_unique = 0; // in case we want to track how much words per local reducer

    /* Start the parallel region! */
    /* Setup our parallel region w/ threads = num_reducers! */
    #pragma omp parallel num_threads(num_reducers)
    {
        #pragma omp for reduction(+:local_unique) schedule(static)

        for (int i = 0; i < num_reducers; i++){ // i -> thread id!
            auto &map = local_maps[i];

            /* Loop through the map and reduce when possible! */
            while (true){
                auto item = reducer_q[i].rb_pop();
                if (item.second == -1) break;
                if (!item.first.empty()) map[item.first] += item.second;
            }

            /* Reached here? Reduction for this local reducer */
            /* is complete! It'll write out everything to the outfile! */
            string outfile = "out_rank" + to_string(rank) + "_r" + to_string(i) + ".txt";
            ofstream fout(outfile);

            for (auto &p : map){
                fout << p.first << " , " << p.second << "\n";
            }

            fout.close();

            local_unique += (int)map.size();

        }
    }

    t_reducer_end = MPI_Wtime(); // Conclude this phase!!
    MPI_Barrier(MPI_COMM_WORLD);

    double t_total_end = MPI_Wtime();

    /* Printing timeeeeee!! */
    printf("[rank %d] timings:\n", rank);
    printf("  Reader   : %.6f s (threads=%d)\n", t_reader_end - t_reader_start, num_readers);
    printf("  Mapper   : %.6f s (threads=%d)\n", t_mapper_end - t_mapper_start, num_mappers);
    printf("  Comm/MPI : %.6f s\n", t_comm_end - t_comm_start);
    printf("  Reducer+Writer : %.6f s (threads=%d)\n", t_reducer_end - t_reducer_start, num_reducers);
    printf("  TOTAL    : %.6f s\n", t_total_end - t_total_start);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;

}
