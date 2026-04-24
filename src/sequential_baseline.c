#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MIN_LAT -90.0
#define MAX_LAT 90.0
#define MIN_LON -180.0
#define MAX_LON 180.0
#define CELL_SIZE 1.0

#define LAT_CELLS ((int)((MAX_LAT - MIN_LAT) / CELL_SIZE))
#define LON_CELLS ((int)((MAX_LON - MIN_LON) / CELL_SIZE))

#define K_CLUSTERS 3
#define MAX_KMEANS_ITERS 100

#define LINE_LEN 512

// Represents a grid cell that consists of a number of earthquakes and their total magnitude.
typedef struct {
    int count;
    double magSum;
} Cell;

// Represents a non-empty grid cell that consists of a number of earthquakes and their average magnitude.
typedef struct {
    int i, j;
    int count;
    double avgMag;
    int clusterId;
} Feature;

// Converts latitude to a grid cell index.
static int latToIndex(double lat) {
    if (lat < MIN_LAT || lat >= MAX_LAT) return -1;
    return (int)((lat - MIN_LAT) / CELL_SIZE);
}

// Converts longitude to a grid cell index.
static int lonToIndex(double lon) {
    if (lon < MIN_LON || lon >= MAX_LON) return -1;
    return (int)((lon - MIN_LON) / CELL_SIZE);
}

// Reads each line from the CSV file and extracts latitude, longitude, and magnitude.
static int parseEarthquake(const char *line, double *lat, double *lon, double *mag) {
    char buf[LINE_LEN];
    strncpy(buf, line, LINE_LEN);
    buf[LINE_LEN - 1] = '\0';

    char *token;
    int col = 0;
    double tLat = 0.0, tLon = 0.0, tMag = 0.0;

    token = strtok(buf, ",");
    while (token != NULL) {
        if (col == 1) {
            tLat = atof(token);
        } else if (col == 2) {
            tLon = atof(token);
        } else if (col == 4) {
            tMag = atof(token);
        }
        col++;
        token = strtok(NULL, ",");
    }

    if (col < 5) return 0;

    *lat = tLat;
    *lon = tLon;
    *mag = tMag;
    return 1;
}

// Computes how different or similar two grid cells are based on number of earthquakes and average magnitude.
static double distanceSquared(double c1, double m1, double c2, double m2) {
    double dc = c1 - c2;
    double dm = m1 - m2;
    return dc * dc + dm * dm;
}

// Groups grid cells into k clusters based on earthquake count and average magnitude.
// (k = 3 in our case: quiet, moderate, hotspot)
static void kMeansCluster(Feature *features, int nFeatures, int k) {
    if (nFeatures <= 0 || k <= 0) return;
    if (k > nFeatures) k = nFeatures;

    double *centCount = (double *)malloc(k * sizeof(double));
    double *centMag   = (double *)malloc(k * sizeof(double));
    int *assignments  = (int *)malloc(nFeatures * sizeof(int));
    if (!centCount || !centMag || !assignments) {
        fprintf(stderr, "K-Means memory allocation failed\n");
        free(centCount);
        free(centMag);
        free(assignments);
        return;
    }

    for (int c = 0; c < k; c++) {
        centCount[c] = (double)features[c].count;
        centMag[c]   = features[c].avgMag;
    }

    for (int iter = 0; iter < MAX_KMEANS_ITERS; iter++) {
        int changed = 0;

        for (int i = 0; i < nFeatures; i++) {
            double pc = (double)features[i].count;
            double pm = features[i].avgMag;
            double bestDist = -1.0;
            int bestC = 0;

            for (int c = 0; c < k; c++) {
                double d2 = distanceSquared(pc, pm, centCount[c], centMag[c]);
                if (bestDist < 0 || d2 < bestDist) {
                    bestDist = d2;
                    bestC = c;
                }
            }

            if (iter == 0 || assignments[i] != bestC) {
                changed++;
                assignments[i] = bestC;
            }
        }

        if (changed == 0) break;

        double *sumC = (double *)calloc(k, sizeof(double));
        double *sumM = (double *)calloc(k, sizeof(double));
        int *cnt     = (int *)calloc(k, sizeof(int));
        if (!sumC || !sumM || !cnt) {
            fprintf(stderr, "K-Means centroid allocation failed\n");
            free(sumC);
            free(sumM);
            free(cnt);
            break;
        }

        for (int i = 0; i < nFeatures; i++) {
            int c = assignments[i];
            sumC[c] += (double)features[i].count;
            sumM[c] += features[i].avgMag;
            cnt[c]  += 1;
        }

        for (int c = 0; c < k; c++) {
            if (cnt[c] > 0) {
                centCount[c] = sumC[c] / cnt[c];
                centMag[c]   = sumM[c] / cnt[c];
            }
        }

        free(sumC);
        free(sumM);
        free(cnt);
    }

    for (int i = 0; i < nFeatures; i++) {
        features[i].clusterId = assignments[i];
    }

    printf("\nCluster centroids (count, avgMag):\n");
    for (int c = 0; c < k; c++) {
        printf("• Cluster %d: count_center=%.2f, avgMag_center=%.2f\n",
               c, centCount[c], centMag[c]);
    }

    free(centCount);
    free(centMag);
    free(assignments);
}

// Labels cluster as either: quiet, moderate, or hotspot based on activity level (earthquake count).
static void labelClusters(Feature *features, int nFeatures, int k,
                          int *clusterRank) {
    double *sumCount = (double *)calloc(k, sizeof(double));
    int *cnt         = (int *)calloc(k, sizeof(int));
    if (!sumCount || !cnt) {
        fprintf(stderr, "Cluster labeling allocation failed\n");
        free(sumCount);
        free(cnt);
        return;
    }

    for (int i = 0; i < nFeatures; i++) {
        int c = features[i].clusterId;
        if (c >= 0 && c < k) {
            sumCount[c] += (double)features[i].count;
            cnt[c]++;
        }
    }

    double *avgCount = (double *)malloc(k * sizeof(double));
    for (int c = 0; c < k; c++) {
        if (cnt[c] > 0) avgCount[c] = sumCount[c] / cnt[c];
        else avgCount[c] = 0.0;
    }

    int *used = (int *)calloc(k, sizeof(int));
    for (int r = 0; r < k; r++) {
        double bestVal = -1.0;
        int bestC = -1;
        for (int c = 0; c < k; c++) {
            if (!used[c]) {
                if (bestC == -1 || avgCount[c] < bestVal) {
                    bestVal = avgCount[c];
                    bestC = c;
                }
            }
        }
        clusterRank[r] = bestC;
        used[bestC] = 1;
    }

    free(sumCount);
    free(cnt);
    free(avgCount);
    free(used);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.csv\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    Cell **grid = (Cell **)malloc(LAT_CELLS * sizeof(Cell *));
    if (!grid) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return 1;
    }
    for (int i = 0; i < LAT_CELLS; i++) {
        grid[i] = (Cell *)calloc(LON_CELLS, sizeof(Cell));
        if (!grid[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(fp);
            return 1;
        }
    }

    char line[LINE_LEN];
    if (fgets(line, LINE_LEN, fp) == NULL) {
        fprintf(stderr, "Empty file\n");
        return 1;
    }

    long totalLines = 0;
    while (fgets(line, LINE_LEN, fp) != NULL) {
        double lat, lon, mag;
        if (!parseEarthquake(line, &lat, &lon, &mag)) {
            continue;
        }
        int i = latToIndex(lat);
        int j = lonToIndex(lon);
        if (i < 0 || j < 0 || i >= LAT_CELLS || j >= LON_CELLS) {
            continue;
        }
        grid[i][j].count += 1;
        grid[i][j].magSum += mag;
        totalLines++;
    }
    fclose(fp);
    printf("Total lines processed: %ld\n", totalLines);

    int maxCells = LAT_CELLS * LON_CELLS;
    Feature *features = (Feature *)malloc(maxCells * sizeof(Feature));
    int featureCount = 0;
    for (int i = 0; i < LAT_CELLS; i++) {
        for (int j = 0; j < LON_CELLS; j++) {
            if (grid[i][j].count > 0) {
                Feature f;
                f.i = i;
                f.j = j;
                f.count = grid[i][j].count;
                f.avgMag = grid[i][j].magSum / grid[i][j].count;
                f.clusterId = -1;
                features[featureCount++] = f;
            }
        }
    }
    printf("Non-empty cells: %d\n", featureCount);

    // Measure only kMeansCluster time
    clock_t kmeans_start = clock();
    kMeansCluster(features, featureCount, K_CLUSTERS);
    clock_t kmeans_end = clock();
    double kmeans_ms = (double)(kmeans_end - kmeans_start) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n===========================================\n");
    printf("Sequential K-Means function execution time: %.3f ms\n", kmeans_ms);
    printf("===========================================\n");
    // -----------------------------------------

    int clusterRank[K_CLUSTERS];
    labelClusters(features, featureCount, K_CLUSTERS, clusterRank);

    const char *labelForCluster[K_CLUSTERS];
    for (int c = 0; c < K_CLUSTERS; c++) {
        labelForCluster[c] = "unknown";
    }
    if (K_CLUSTERS >= 1) labelForCluster[clusterRank[0]] = "quiet";
    if (K_CLUSTERS >= 2) labelForCluster[clusterRank[1]] = "moderate";
    if (K_CLUSTERS >= 3) labelForCluster[clusterRank[2]] = "hotspot";

    printf("\nCluster ranking by activity:\n");
    for (int r = 0; r < K_CLUSTERS; r++) {
        int c = clusterRank[r];
        const char *lab = labelForCluster[c];
        printf("• Rank %d: Cluster %d (%s)\n", r, c, lab);
    }

    printf("\nHotspot map (one line per non-empty cell):\n");
    printf("lat_center,lon_center,count,avgMag,clusterId,clusterLabel\n");
    for (int k = 0; k < featureCount; k++) {
        Feature *f = &features[k];
        double cellLat = MIN_LAT + (f->i + 0.5) * CELL_SIZE;
        double cellLon = MIN_LON + (f->j + 0.5) * CELL_SIZE;
        int cid = f->clusterId;
        const char *lab = (cid >= 0 && cid < K_CLUSTERS)
                          ? labelForCluster[cid]
                          : "unknown";
        printf("%.3f,%.3f,%d,%.3f,%d,%s\n",
               cellLat, cellLon, f->count, f->avgMag, cid, lab);
    }

    for (int i = 0; i < LAT_CELLS; i++) {
        free(grid[i]);
    }
    free(grid);
    free(features);

    return 0;
}
