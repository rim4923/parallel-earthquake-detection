#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MIN_LAT -90.0
#define MAX_LAT  90.0
#define MIN_LON -180.0
#define MAX_LON  180.0
#define CELL_SIZE 1.0

#define LAT_CELLS ((int)((MAX_LAT - MIN_LAT) / CELL_SIZE))
#define LON_CELLS ((int)((MAX_LON - MIN_LON) / CELL_SIZE))

#define MAX_LINE 512
#define MAX_KMEANS_ITERS 100
#define K 3

typedef struct {
    int count;
    double sumMag;
    double avgMag;
    int clusterId;
} Cell;

typedef struct {
    double count;
    double avgMag;
} Feature;

Cell **alloc_grid() {
    Cell **grid = malloc(LAT_CELLS * sizeof(Cell *));
    for (int i = 0; i < LAT_CELLS; i++)
        grid[i] = calloc(LON_CELLS, sizeof(Cell));
    return grid;
}

void free_grid(Cell **grid) {
    for (int i = 0; i < LAT_CELLS; i++) free(grid[i]);
    free(grid);
}

int latlon_to_idx(double lat, double lon, int *ilat, int *ilon) {
    if (lat < MIN_LAT || lat >= MAX_LAT) return 0;
    if (lon < MIN_LON || lon >= MAX_LON) return 0;
    *ilat = (int)((lat - MIN_LAT) / CELL_SIZE);
    *ilon = (int)((lon - MIN_LON) / CELL_SIZE);
    return (*ilat >= 0 && *ilat < LAT_CELLS && *ilon >= 0 && *ilon < LON_CELLS);
}

void process_csv(const char *file, Cell **grid, long *total) {
    FILE *fp = fopen(file, "r");
    char line[MAX_LINE];
    fgets(line, sizeof(line), fp);
    long lines = 0;

    while (fgets(line, sizeof(line), fp)) {
        lines++;
        char *tmp = strdup(line);
        char *tok = strtok(tmp, ",");
        int col = 0;
        double lat = 0, lon = 0, mag = 0;
        while (tok) {
            if (col == 1) lat = atof(tok);
            else if (col == 2) lon = atof(tok);
            else if (col == 4) mag = atof(tok);
            tok = strtok(NULL, ",");
            col++;
        }
        free(tmp);

        int ilat, ilon;
        if (latlon_to_idx(lat, lon, &ilat, &ilon)) {
            Cell *c = &grid[ilat][ilon];
            c->count++;
            c->sumMag += mag;
        }
    }
    fclose(fp);

    for (int i = 0; i < LAT_CELLS; i++)
        for (int j = 0; j < LON_CELLS; j++)
            if (grid[i][j].count > 0)
                grid[i][j].avgMag = grid[i][j].sumMag / grid[i][j].count;

    *total = lines;
}

Feature *build_features(Cell **grid, int *n) {
    int count = 0;
    for (int i = 0; i < LAT_CELLS; i++)
        for (int j = 0; j < LON_CELLS; j++)
            if (grid[i][j].count > 0) count++;

    *n = count;
    Feature *f = malloc(count * sizeof(Feature));
    int idx = 0;

    for (int i = 0; i < LAT_CELLS; i++)
        for (int j = 0; j < LON_CELLS; j++)
            if (grid[i][j].count > 0) {
                f[idx].count = grid[i][j].count;
                f[idx].avgMag = grid[i][j].avgMag;
                idx++;
            }

    return f;
}

void mpi_kmeans(Feature *feat, int n, int k,
                int *assign, int rank, int size) {

    double *centC = malloc(k * sizeof(double));
    double *centM = malloc(k * sizeof(double));

    if (rank == 0)
        for (int c = 0; c < k; c++) {
            centC[c] = feat[c].count;
            centM[c] = feat[c].avgMag;
        }

    MPI_Bcast(centC, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centM, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int base = n / size;
    int rem = n % size;
    int local_n = base + (rank < rem);
    int start = rank * base + (rank < rem ? rank : rem);

    int *local_assign = malloc(local_n * sizeof(int));
    double *sumC_l = calloc(k, sizeof(double));
    double *sumM_l = calloc(k, sizeof(double));
    int *cnt_l = calloc(k, sizeof(int));
    double *sumC_g = calloc(k, sizeof(double));
    double *sumM_g = calloc(k, sizeof(double));
    int *cnt_g = calloc(k, sizeof(int));

    for (int iter = 0; iter < MAX_KMEANS_ITERS; iter++) {
        for (int c = 0; c < k; c++)
            sumC_l[c] = sumM_l[c] = cnt_l[c] = 0;

        int changed_l = 0;

        for (int li = 0; li < local_n; li++) {
            int i = start + li;
            double pc = feat[i].count;
            double pm = feat[i].avgMag;

            double best = 1e300;
            int bestC = 0;

            for (int c = 0; c < k; c++) {
                double d = (pc - centC[c])*(pc - centC[c]) +
                           (pm - centM[c])*(pm - centM[c]);
                if (d < best) best = d, bestC = c;
            }

            if (iter == 0 || local_assign[li] != bestC)
                changed_l++;

            local_assign[li] = bestC;
            sumC_l[bestC] += pc;
            sumM_l[bestC] += pm;
            cnt_l[bestC]++;
        }

        int changed_g;
        MPI_Allreduce(&changed_l, &changed_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(sumC_l, sumC_g, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(sumM_l, sumM_g, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_l, cnt_g, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0)
            for (int c = 0; c < k; c++)
                if (cnt_g[c] > 0) {
                    centC[c] = sumC_g[c] / cnt_g[c];
                    centM[c] = sumM_g[c] / cnt_g[c];
                }

        MPI_Bcast(centC, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(centM, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (changed_g == 0) break;
    }

    int *recvcounts = NULL, *displs = NULL;
    int ln = local_n;

    if (rank == 0) {
        recvcounts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));
    }

    MPI_Gather(&ln, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < size; r++)
            displs[r] = displs[r-1] + recvcounts[r-1];
    }

    MPI_Gatherv(local_assign, local_n, MPI_INT,
                assign, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    free(local_assign);
    free(sumC_l); free(sumM_l); free(cnt_l);
    free(sumC_g); free(sumM_g); free(cnt_g);
    if (rank == 0) { free(recvcounts); free(displs); }
}

void write_results(const char *file, Cell **grid, int k,
                   int *assign, int nFeat, long total) {

    FILE *fp = fopen(file, "a");
    double sumC[K]={0}, sumM[K]={0};
    int cnt[K]={0};

    int idx=0;
    for (int i=0;i<LAT_CELLS;i++)
        for (int j=0;j<LON_CELLS;j++)
            if (grid[i][j].count>0){
                int cid=assign[idx];
                grid[i][j].clusterId=cid;
                sumC[cid]+=grid[i][j].count;
                sumM[cid]+=grid[i][j].avgMag;
                cnt[cid]++;
                idx++;
            }

    double centC[K], centM[K];
    for (int c=0;c<k;c++){
        centC[c]=(cnt[c]>0?sumC[c]/cnt[c]:0);
        centM[c]=(cnt[c]>0?sumM[c]/cnt[c]:0);
    }

    int rnk[K]={0,1,2};
    for (int i=0;i<k-1;i++)
        for (int j=0;j<k-i-1;j++)
            if (centC[rnk[j]]>centC[rnk[j+1]]){
                int t=rnk[j]; rnk[j]=rnk[j+1]; rnk[j+1]=t;
            }

    const char *labname[3]={"quiet","moderate","hotspot"};
    const char *lab[K];
    for(int c=0;c<k;c++) lab[c]=labname[rnk[c]];

    fprintf(fp,"Total lines processed: %ld\n",total);

    int non=0;
    for(int i=0;i<LAT_CELLS;i++)
        for(int j=0;j<LON_CELLS;j++)
            if(grid[i][j].count>0) non++;

    fprintf(fp,"Non-empty cells: %d\n\n",non);
    fprintf(fp,"Cluster centroids (count,avgMag):\n");
    for(int c=0;c<k;c++)
        fprintf(fp,"Cluster %d: %.2f %.2f\n",c,centC[c],centM[c]);

    fprintf(fp,"\nCluster ranking:\n");
    for(int r=0;r<k;r++){
        int cid=rnk[r];
        fprintf(fp,"Rank %d: Cluster %d (%s)\n",r,cid,lab[cid]);
    }

    fprintf(fp,"\nHotspot map:\n");
    fprintf(fp,"lat,lon,count,avgMag,cluster,label\n");

    for (int i=0;i<LAT_CELLS;i++)
        for (int j=0;j<LON_CELLS;j++)
            if(grid[i][j].count>0){
                double lat=MIN_LAT+(i+0.5)*CELL_SIZE;
                double lon=MIN_LON+(j+0.5)*CELL_SIZE;
                int cid=grid[i][j].clusterId;
                fprintf(fp,"%.3f,%.3f,%d,%.3f,%d,%s\n",
                        lat,lon,
                        grid[i][j].count,
                        grid[i][j].avgMag,
                        cid,lab[cid]);
            }

    fclose(fp);
}

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    Cell **grid=alloc_grid();
    Feature *features=NULL;
    int nf=0;
    long total=0;
    double t0=0;

    if(rank==0){
        t0=MPI_Wtime();
        const char *file=(argc>1?argv[1]:"EarthquakeData1930To2025.csv");
        process_csv(file,grid,&total);
        features=build_features(grid,&nf);
    }

    MPI_Bcast(&nf,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank!=0) features=malloc(nf*sizeof(Feature));

    MPI_Datatype FT;
    int bl[2]={1,1};
    MPI_Aint disp[2],base;
    Feature f0;
    MPI_Get_address(&f0,&base);
    MPI_Get_address(&f0.count,&disp[0]);
    MPI_Get_address(&f0.avgMag,&disp[1]);
    disp[0]-=base; disp[1]-=base;
    MPI_Datatype tp[2]={MPI_DOUBLE,MPI_DOUBLE};
    MPI_Type_create_struct(2,bl,disp,tp,&FT);
    MPI_Type_commit(&FT);

    MPI_Bcast(features,nf,FT,0,MPI_COMM_WORLD);

    int *assign=NULL;
    if(rank==0) assign=malloc(nf*sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    double k0=MPI_Wtime();

    mpi_kmeans(features,nf,K,assign,rank,size);

    MPI_Barrier(MPI_COMM_WORLD);
    double k1=MPI_Wtime();

    if(rank==0){
        double t1=MPI_Wtime();
        double k_ms=(k1-k0)*1000.0;
        double tot_ms=(t1-t0)*1000.0;

        char outfile[64];
        sprintf(outfile, "mpi_results_%d.csv", size);

        FILE *fp=fopen(outfile,"w");
        fprintf(fp,"KMeans_ms: %.3f\n",k_ms);
        fprintf(fp,"TotalPipeline_ms: %.3f\n\n",tot_ms);
        fclose(fp);

        write_results(outfile,grid,K,assign,nf,total);

        printf("Saved results to %s\n", outfile);
    }

    MPI_Type_free(&FT);
    free_grid(grid);
    free(features);
    if(rank==0) free(assign);
    MPI_Finalize();
    return 0;
}