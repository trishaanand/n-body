/*  
    N-Body simulation code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>


#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015


struct bodyType {
    double x1[MAXBODIES];        /* Old and new X-axis coordinates */
    double x2[MAXBODIES];
    double y1[MAXBODIES];        /* Old and new Y-axis coordinates */
    double y2[MAXBODIES];
    double xf[MAXBODIES];          /* force along X-axis */
    double yf[MAXBODIES];          /* force along Y-axis */
    double xv[MAXBODIES];          /* velocity along X-axis */
    double yv[MAXBODIES];          /* velocity along Y-axis */
    double mass[MAXBODIES];        /* Mass of the body */
    double radius[MAXBODIES];      /* width (derived from mass) */
    double *x_old, *x_new, *y_old, *y_new;
};


struct world {
    struct bodyType bodies;
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
    
};

/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies.x_old[B]
#define XN(w, B)       (w)->bodies.x_new[B]
#define Y(w, B)        (w)->bodies.y_old[B]
#define YN(w, B)       (w)->bodies.y_new[B]
#define XF(w, B)       (w)->bodies.xf[B]
#define YF(w, B)       (w)->bodies.yf[B]
#define XV(w, B)       (w)->bodies.xv[B]
#define YV(w, B)       (w)->bodies.yv[B]
#define R(w, B)        (w)->bodies.radius[B]
#define M(w, B)        (w)->bodies.mass[B]


static void
clear_forces(struct world *world, int low, int high)
{
    int b;

    /* Clear force accumulation variables */
    for (b = low; b <= high; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

static void
compute_forces(struct world *world, int low, int high)
{
    int b, c;

    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    for (b = low; b <= high; ++b) {
        // for (c = b + 1; c < world->bodyCt; ++c) {
        for (c = 0; c < world->bodyCt; ++c) {
            if (b != c) {
            double dx = X(world, c) - X(world, b);
            double dy = Y(world, c) - Y(world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(world, b) + R(world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(world, b) * M(world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            XF(world, b) += xf;
            YF(world, b) += yf;

            }
        }
    }
}

static void
compute_velocities(struct world *world, int low, int high)
{
    int b;

    for (b = low; b <= high; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;
    }
}

static void
compute_positions(struct world *world, int low, int high)
{
    int b;

    for (b = 0; b <= high; ++b) {
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;
    }
}


/*  Graphic output stuff...
 */

#include <fcntl.h>
#include <sys/mman.h>

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void
filemap_close(struct filemap *filemap)
{
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *
Eat_Space(unsigned char *p)
{
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }

    return p;
}


static unsigned char *
Get_Number(unsigned char *p, int *n)
{
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int
map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap)
{
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }

    /* File should now be mapped; read magic value */
    p = filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;

    return 0;

ppm_abort:
    filemap_close(filemap);

    return -1;
}


static inline void
color(const struct world *world, unsigned char *image, int x, int y, int b)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));
    int tint = ((0xfff * (b + 1)) / (world->bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(const struct world *world, unsigned char *image, int x, int y)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

static void
display(const struct world *world, unsigned char *image)
{
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < world->ydim; ++j) {
        for (i = 0; i < world->xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < world->bodyCt; ++b) {
                double dy = Y(world, b) - j;
                double dx = X(world, b) - i;
                double d = sqrt(dx*dx + dy*dy);

                if (d <= R(world, b)+0.5) {
                    /* This is it */
                    color(world, image, i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(world, image, i, j);

colored:        ;
        }
    }
}

static void
print(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
    }
}

void local_low_high_indices(int process_id, int *count_per_process, int num_processes, int *low, int *high) {
    int i = 0;
    int tmp_low = 0;
    while(i < process_id) {
        tmp_low += count_per_process[i];
        i++;
    }
    *low = tmp_low;
    *high = tmp_low + count_per_process[i] - 1; 
    // fprintf(stderr, "In process %d, low is %d, high is %d, count_per_process")
}
void
do_compute(int steps, struct world *world, int process_id, int num_processes, int *count_per_process, int *displs_array) {
    //local index
    int low, high = 0;
    local_low_high_indices(process_id, count_per_process, num_processes, &low, &high);
    fprintf(stderr, "In process %d, low is %d, high is %d\n", process_id, low, high);
    
    while (steps--) {
        if (process_id == 0) {
            clear_forces(world, low, high);
            compute_forces(world, low, high);
            compute_velocities(world, low, high);
            compute_positions(world, low, high);

            /* Flip old & new coordinates */
            world->old ^= 1;
            // double *tmp = world->bodies.x_old;
            world->bodies.x_old = world->bodies.x_new;
            // world->bodies.x_new = tmp;

            // tmp = world->bodies.y_old;
            world->bodies.y_old = world->bodies.y_new;
            // world->bodies.y_new = tmp;
        }

        // /*Time for a display update?*/ 
        // if (secsup > 0 && (time(0) - lastup) > secsup) {
        //     display(world, image_map.image);
        //     msync(image_map.map, image_map.fsize, MS_SYNC); /* Force write */
        //     lastup = time(0);
        // }
        // fprintf(stderr, "In process %d, low is %d, count/process is  %d\n", process_id, low, count_per_process[process_id]);
        MPI_Allgatherv(&world->bodies.x_old[low], count_per_process[process_id], MPI_DOUBLE,
                    world->bodies.x_old, count_per_process, displs_array, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(&world->bodies.y_old[low], count_per_process[process_id], MPI_DOUBLE,
                    world->bodies.y_old, count_per_process, displs_array, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Allgatherv(&world->bodies.xf[low], count_per_process[process_id], MPI_DOUBLE, world->bodies.xf,
                count_per_process, displs_array, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(&world->bodies.yf[low], count_per_process[process_id], MPI_DOUBLE, world->bodies.yf,
                count_per_process, displs_array, MPI_DOUBLE, MPI_COMM_WORLD);

}

void send_receive_initial_data(struct world *world, int process_id, int num_processes) 
{
    int world_size = world->bodyCt;

    MPI_Bcast (world->bodies.mass, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (world->bodies.radius, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);


}

/*  Main program...
*/
int
main(int argc, char **argv)
{   
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // Get the rank of the process
    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    // fprintf(stderr, "Rank : %d out of %d\n", process_id, num_processes);

    // unsigned int lastup = 0;
    // unsigned int secsup;
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
    struct filemap image_map;
    struct world *world;
    int i;
    int loc_num = 0;
    int *count_per_process;
    int *displs_array;

    world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    if (argc != 4) {
        fprintf(stderr, "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;
    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }
    // secsup = atoi(argv[2]);
    if (map_P6(argv[2], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }
    steps = atoi(argv[3]);

    // if (process_id == 0) {
        fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);

        /* Initialize simulation data */
        srand(SEED);
        world->bodies.x_old = world->bodies.x1;
        world->bodies.y_old = world->bodies.y1;
        world->bodies.x_new = world->bodies.x2;
        world->bodies.y_new = world->bodies.y2;

        for (b = 0; b < world->bodyCt; ++b) {
            X(world, b) = (rand() % world->xdim);
            Y(world, b) = (rand() % world->ydim);
            R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                    (25.0 * (world->bodyCt*world->bodyCt + 1.0));
            M(world, b) = R(world, b) * R(world, b) * R(world, b);
            XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
            YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        }
    // }

    loc_num = (int) (world->bodyCt/num_processes);
    // fprintf(stderr, "count per process is : %d\n", loc_num);
    // if (world->bodyCt%num_processes != 0) {
    //     loc_num += 1;
    // }
    count_per_process = malloc(num_processes * sizeof(int));
    for (i=0; i<num_processes; i++) {
        if (i < world->bodyCt%num_processes) {
            count_per_process[i] = loc_num + 1;
        } else {
            count_per_process[i] = loc_num;
        }
        // fprintf(stderr, " count per process[%d] is %d\n", i, count_per_process[i]);
    }
    displs_array = malloc(num_processes * sizeof(int));
    int tmp = 0;
    for (i=0; i<num_processes; i++) {
        displs_array[i] = tmp;
        tmp += count_per_process[i];
    }
    send_receive_initial_data(world, process_id ,num_processes);

    if (process_id == 0) {
        if (gettimeofday(&start, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
    }

    /* Main Execution */
    do_compute(steps, world, process_id, num_processes, count_per_process, displs_array);
    
    if (process_id == 0) {
        if (gettimeofday(&end, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
        rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) - 
                    (start.tv_sec + (start.tv_usec / 1000000.0));

        fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
        //the output of world is mixing with the previous fprintf. adding sleep
        // sleep(5);
        print(world);

        filemap_close(&image_map);
    }


    free(world);
    MPI_Finalize();

    return 0;
}
