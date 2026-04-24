#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MIN_LAT  -90.0
#define MAX_LAT   90.0
#define MIN_LON -180.0
#define MAX_LON  180.0
#define CELL_SIZE 1.0

#define LAT_CELLS ((int)((MAX_LAT - MIN_LAT) / CELL_SIZE))
#define LON_CELLS ((int)((MAX_LON - MIN_LON) / CELL_SIZE))

#define K_CLUSTERS 3
#define MAX_KMEANS_ITERS 100
#define LINE_LEN 512

typedef struct {
    int count;
    double magSum;
} Cell;

typedef struct {
    int i, j;
    int count;
    double avgMag;
    int clusterId;
} Feature;

// ---------------- CPU HELPERS ----------------
static int latToIndex(double lat){
    if(lat < MIN_LAT || lat >= MAX_LAT) return -1;
    return (int)((lat - MIN_LAT)/CELL_SIZE);
}

static int lonToIndex(double lon){
    if(lon < MIN_LON || lon >= MAX_LON) return -1;
    return (int)((lon - MIN_LON)/CELL_SIZE);
}

static int parseEarthquake(const char *line, double *lat, double *lon, double *mag){
    char buf[LINE_LEN];
    strncpy(buf, line, LINE_LEN);
    buf[LINE_LEN-1] = '\0';
    char *token = strtok(buf, ",");
    int col = 0;
    double tLat=0, tLon=0, tMag=0;

    while(token != NULL){
        if(col == 1) tLat = atof(token);
        else if(col == 2) tLon = atof(token);
        else if(col == 4) tMag = atof(token);
        col++;
        token = strtok(NULL, ",");
    }

    if(col < 5) return 0;
    *lat = tLat; *lon = tLon; *mag = tMag;
    return 1;
}

// ------------- CUDA HELPERS ---------------
static void checkCuda(cudaError_t err, const char *msg){
    if(err != cudaSuccess){
        printf("CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__device__ float dist2(float c1, float m1, float c2, float m2){
    float dc = c1 - c2;
    float dm = m1 - m2;
    return dc*dc + dm*dm;
}

// ------------------- CUDA KERNEL --------------------
__global__ void kmeansAssignReduce(
    const float *featCount, const float *featMag, int n,
    const float *centCount, const float *centMag, int *assignments,
    float *sumCount, float *sumMag, int *clusterCount, int *changed
){
    extern __shared__ float smem[];
    float *s_centCount = smem;
    float *s_centMag   = s_centCount + K_CLUSTERS;
    float *s_sumCount  = s_centMag   + K_CLUSTERS;
    float *s_sumMag    = s_sumCount  + K_CLUSTERS;
    int   *s_cnt       = (int *)(s_sumMag + K_CLUSTERS);

    int tid = threadIdx.x;

    if(tid < K_CLUSTERS){
        s_centCount[tid] = centCount[tid];
        s_centMag[tid]   = centMag[tid];
        s_sumCount[tid]  = 0;
        s_sumMag[tid]    = 0;
        s_cnt[tid]       = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if(idx < n){
        float fc = featCount[idx];
        float fm = featMag[idx];

        int best = 0;
        float bestDist = dist2(fc, fm, s_centCount[0], s_centMag[0]);
        for(int c=1;c<K_CLUSTERS;c++){
            float d = dist2(fc,fm,s_centCount[c],s_centMag[c]);
            if(d < bestDist){
                bestDist = d;
                best = c;
            }
        }

        if(assignments[idx] != best){
            atomicAdd(changed, 1);
            assignments[idx] = best;
        }

        atomicAdd(&s_sumCount[best], fc);
        atomicAdd(&s_sumMag[best], fm);
        atomicAdd(&s_cnt[best], 1);
    }
    __syncthreads();

    if(tid < K_CLUSTERS){
        if(s_cnt[tid] > 0){
            atomicAdd(&sumCount[tid], s_sumCount[tid]);
            atomicAdd(&sumMag[tid],   s_sumMag[tid]);
            atomicAdd(&clusterCount[tid], s_cnt[tid]);
        }
    }
}

__global__ void kmeansRecompute(const float *sumCount, const float *sumMag,
                                const int *clusterCount,
                                float *centCount, float *centMag){
    int c = threadIdx.x;
    if(c < K_CLUSTERS && clusterCount[c] > 0){
        centCount[c] = sumCount[c] / clusterCount[c];
        centMag[c]   = sumMag[c]   / clusterCount[c];
    }
}

// ------------------- GPU KMEANS ---------------------
void runKMeansCUDA(Feature *features, int nFeatures, float *gpu_ms_out, int blockSize){
    float *h_fc = (float*)malloc(nFeatures*sizeof(float));
    float *h_fm = (float*)malloc(nFeatures*sizeof(float));
    int   *h_assign = (int*)malloc(nFeatures*sizeof(int));
    float h_centCount[K_CLUSTERS], h_centMag[K_CLUSTERS];

    for(int i=0;i<nFeatures;i++){
        h_fc[i] = features[i].count;
        h_fm[i] = features[i].avgMag;
        h_assign[i] = -1;
    }

    for(int c=0;c<K_CLUSTERS;c++){
        h_centCount[c] = features[c].count;
        h_centMag[c]   = features[c].avgMag;
    }

    float *d_fc, *d_fm, *d_centCount, *d_centMag;
    float *d_sumCount, *d_sumMag;
    int *d_assign, *d_cnt, *d_changed;

    cudaMalloc(&d_fc, nFeatures*sizeof(float));
    cudaMalloc(&d_fm, nFeatures*sizeof(float));
    cudaMalloc(&d_assign, nFeatures*sizeof(int));
    cudaMalloc(&d_centCount, K_CLUSTERS*sizeof(float));
    cudaMalloc(&d_centMag,   K_CLUSTERS*sizeof(float));
    cudaMalloc(&d_sumCount,  K_CLUSTERS*sizeof(float));
    cudaMalloc(&d_sumMag,    K_CLUSTERS*sizeof(float));
    cudaMalloc(&d_cnt,       K_CLUSTERS*sizeof(int));
    cudaMalloc(&d_changed,   sizeof(int));

    cudaMemcpy(d_fc, h_fc, nFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fm, h_fm, nFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assign, h_assign, nFeatures*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centCount, h_centCount, K_CLUSTERS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centMag,   h_centMag,   K_CLUSTERS*sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (nFeatures + blockSize - 1) / blockSize;
    size_t shmem = (K_CLUSTERS*4*sizeof(float)) + (K_CLUSTERS*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int iter=0;iter<MAX_KMEANS_ITERS;iter++){
        cudaMemset(d_sumCount,0,K_CLUSTERS*sizeof(float));
        cudaMemset(d_sumMag,0,K_CLUSTERS*sizeof(float));
        cudaMemset(d_cnt,0,K_CLUSTERS*sizeof(int));
        cudaMemset(d_changed,0,sizeof(int));

        kmeansAssignReduce<<<gridSize,blockSize,shmem>>>(
            d_fc,d_fm,nFeatures,
            d_centCount,d_centMag,
            d_assign,
            d_sumCount,d_sumMag,d_cnt,
            d_changed
        );
        cudaDeviceSynchronize();

        kmeansRecompute<<<1,K_CLUSTERS>>>(d_sumCount,d_sumMag,d_cnt,d_centCount,d_centMag);
        cudaDeviceSynchronize();

        int h_change;
        cudaMemcpy(&h_change,d_changed,sizeof(int),cudaMemcpyDeviceToHost);
        if(h_change == 0) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(gpu_ms_out,start,stop);

    cudaMemcpy(h_assign, d_assign, nFeatures*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centCount, d_centCount, K_CLUSTERS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centMag,   d_centMag,   K_CLUSTERS*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<nFeatures;i++)
        features[i].clusterId = h_assign[i];

    cudaFree(d_fc); cudaFree(d_fm); cudaFree(d_assign);
    cudaFree(d_centCount); cudaFree(d_centMag);
    cudaFree(d_sumCount); cudaFree(d_sumMag);
    cudaFree(d_cnt); cudaFree(d_changed);
    free(h_fc); free(h_fm); free(h_assign);

    printf("\nCUDA K-Means centroids:\n");
    for(int c=0;c<K_CLUSTERS;c++)
        printf("Cluster %d → count_center=%.2f, avgMag_center=%.2f\n",
               c, h_centCount[c], h_centMag[c]);
}

// ------------------- MAIN --------------------
int main(int argc, char *argv[]){
    if(argc < 2){
        printf("Usage: %s <csv> [blockSize]\n", argv[0]);
        return 1;
    }

    int blockSize = 256;
    if(argc >= 3) blockSize = atoi(argv[2]);

    clock_t start_total = clock();

    // Read CSV & build grid EXACTLY LIKE SEQUENTIAL
    FILE *fp = fopen(argv[1],"r");
    if(!fp){ printf("Cannot open file.\n"); return 1; }

    Cell **grid = (Cell**)malloc(LAT_CELLS*sizeof(Cell*));
    for(int i=0;i<LAT_CELLS;i++)
        grid[i] = (Cell*)calloc(LON_CELLS,sizeof(Cell));

    char line[LINE_LEN];
    fgets(line,LINE_LEN,fp); // skip header
    long count=0;

    while(fgets(line,LINE_LEN,fp)){
        double la,lo,ma;
        if(!parseEarthquake(line,&la,&lo,&ma)) continue;

        int I=latToIndex(la), J=lonToIndex(lo);
        if(I<0||J<0||I>=LAT_CELLS||J>=LON_CELLS) continue;

        grid[I][J].count++;
        grid[I][J].magSum += ma;
        count++;
    }
    fclose(fp);

    printf("Earthquakes used: %ld\n", count);

    Feature *features = (Feature*)malloc(LAT_CELLS*LON_CELLS*sizeof(Feature));
    int nF=0;

    for(int i=0;i<LAT_CELLS;i++){
        for(int j=0;j<LON_CELLS;j++){
            if(grid[i][j].count>0){
                features[nF].i=i;
                features[nF].j=j;
                features[nF].count=grid[i][j].count;
                features[nF].avgMag=grid[i][j].magSum/grid[i][j].count;
                features[nF].clusterId=-1;
                nF++;
            }
        }
    }

    printf("Non-empty cells (features): %d\n", nF);

    float kmeans_ms = 0.0f;
    runKMeansCUDA(features,nF,&kmeans_ms,blockSize);

    clock_t end_total = clock();
    double total_ms = (double)(end_total-start_total)*1000/CLOCKS_PER_SEC;

    printf("\n========== RESULTS ==========\n");
    printf("CUDA K-Means time: %.3f ms\n", kmeans_ms);
    printf("Total CUDA program time: %.3f ms\n", total_ms);
    printf("Block size used: %d\n", blockSize);
    printf("==============================\n");

    return 0;
}
