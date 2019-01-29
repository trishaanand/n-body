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
clear_forces(struct world *world, int *particle_distribution, int total)
{
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < total; ++b) {
        YF(world, particle_distribution[b]) = XF(world, particle_distribution[b]) = 0;
    }
}

static void compute_force_two_particles(struct world *world, int b, double x1, double y1, int c, double x2, double y2, double *xf, double *yf) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double angle = atan2(dy, dx);
    double dsqr = dx*dx + dy*dy;
    double mindist = R(world, b) + R(world, c);
    double mindsqr = mindist*mindist;
    double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
    double force = M(world, b) * M(world, c) * GRAVITY / forced;
    *xf = force * cos(angle);
    *yf = force * sin(angle);

}

static void
compute_forces(struct world *world, int *particle_distribution, int total, int max_total_num, double *tmp_comm_array, 
                int tmp_array_size, int process_id, int num_processes)
{   
    MPI_Status status;
    int i,j;
    
    //Pointers to point to the particle indexes and forces in the tmp_comm_array
    double *index, *fx, *fy, *x, *y;
    index = &tmp_comm_array[1];
    fx = &tmp_comm_array[1 + max_total_num];
    fy = &tmp_comm_array[1 + 2*max_total_num];
    x = &tmp_comm_array[1 + 3*max_total_num];
    y = &tmp_comm_array[1 + 4*max_total_num];
    
    //Fill the tmp_comm_array with all the values from this process.
    memset(tmp_comm_array, 0, sizeof(double)*tmp_array_size);
    tmp_comm_array[0] = total;
    
    for (i = 0; i < total; i++) {
        index[i] = (double)particle_distribution[i];
        x[i] = X(world, particle_distribution[i]);
        y[i] = Y(world, particle_distribution[i]);
    }

    int tmp_tot = 0;

    //Recv particles from left, send to the right 
    int neighbour_recv = (process_id - 1 + num_processes) % num_processes;
    int neighbour_send = (process_id + 1) % num_processes;
    int round = 0;
    // ----------------------------------------
    // for (round = 0; round < num_processes ; round ++) {
    do {
        tmp_tot = (int)tmp_comm_array[0];
        for (i = 0; i < total; i++) {
            int own_particle = particle_distribution[i];
            double x1 = X(world, own_particle);
            double y1 = Y(world, own_particle);
            for (j = 0; j < tmp_tot; j++) {
                double xf, yf;
                int idx_int = (int) index[j];
                if (own_particle < idx_int) {
                    double x2 = x[j];
                    double y2 = y[j];
                    
                    compute_force_two_particles(world, own_particle, x1, y1,
                                                idx_int, x2, y2, &xf, &yf);
                    XF(world, own_particle) += xf;
                    YF(world, own_particle) += yf;

                    fx[j] -= xf;
                    fy[j] -= yf;
                }
            }
        }

        //Send receive tmp_comm_array from neighbours
        MPI_Sendrecv_replace(tmp_comm_array, tmp_array_size, MPI_DOUBLE, neighbour_send, 0, 
                            neighbour_recv, 0, MPI_COMM_WORLD, &status);
        round++;
    } while (round < num_processes);
    // MPI_Sendrecv_replace(tmp_comm_array, tmp_array_size, MPI_DOUBLE, neighbour_send, 0, 
    //                         neighbour_recv, 0, MPI_COMM_WORLD, &status);
        

    // Add all the forces from the tmp_comm_array to world forces of locally owned particles
    int par = 0;
    for (i=0; i<tmp_tot; i++) {
        par = (int) index[i];
        XF(world, par) += fx[i];
        YF(world, par) += fy[i];
    }

}

static void
compute_velocities(struct world *world, int *particle_distribution, int total)
{
    int b, par;

    for (b = 0; b <total; ++b) {
        par = particle_distribution[b];
        double xv = XV(world, par);
        double yv = YV(world, par);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, par) - (force * cos(angle));
        double yf = YF(world, par) - (force * sin(angle));

        XV(world, par) += (xf / M(world, par)) * DELTA_T;
        YV(world, par) += (yf / M(world, par)) * DELTA_T;
    }
}

static void
compute_positions(struct world *world, int *particle_distribution, int total)
{
    int b, par;

    for (b = 0; b <total; ++b) {
        par = particle_distribution[b];
        double xn = X(world, par) + XV(world, par) * DELTA_T;
        double yn = Y(world, par) + YV(world, par) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, par) = -XV(world, par);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, par) = -XV(world, par);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, par) = -YV(world, par);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, par) = -YV(world, par);
        }

        /* Update position */
        XN(world, par) = xn;
        YN(world, par) = yn;
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

void
do_compute(unsigned int secsup, struct filemap image_map, int steps, struct world *world, int process_id, int num_processes, int *count_per_process) {
    MPI_Status status;
    unsigned int lastup = 0;
    
    int i, j = 0;
    
    /* 
     * Do cyclic distribution of particles for load balancing
     * particle_distribution is the array of particles owned by this particular locale
     */
    int *particle_distribution = malloc (count_per_process[process_id] * sizeof(int));

    for (i = 0 ; i < world->bodyCt; i++) {
        int rem = i % num_processes;
        if (rem != process_id) continue;
        int div = i / num_processes;
        particle_distribution[div] = i;
    }

    // Get the max number of particles on any locale. Required for index referencing later on
    int max_total_num = (int) (world->bodyCt/num_processes);
    if (world->bodyCt % num_processes != 0) {
        max_total_num += 1;
    }

    /*
    Communication array needs to store : 
    1. number of particles, 
    2. array of indexes, 
    3. xforce 
    4. yforce
    5. x_loc 
    6. y_loc
    */ 
    int arr_size = 1 + max_total_num * 5;
    double *tmp_comm_array = malloc (arr_size * sizeof(double));
    if (tmp_comm_array == NULL) {
        fprintf(stderr, "[%d]tmp_comm_array is NULL. Exiting \n", process_id);
    }

    while (steps--) {
        clear_forces(world, particle_distribution, count_per_process[process_id]);
        compute_forces(world, particle_distribution, count_per_process[process_id], max_total_num, tmp_comm_array, arr_size, process_id, num_processes);
        compute_velocities(world, particle_distribution,  count_per_process[process_id]);
        compute_positions(world, particle_distribution,  count_per_process[process_id]);

        /* Flip old & new coordinates */
        world->bodies.x_old = world->bodies.x_new;
        world->bodies.y_old = world->bodies.y_new;

        /*Time for a display update?*/ 
        if (secsup > 0 && (time(0) - lastup) > secsup) {
            display(world, image_map.image);
            msync(image_map.map, image_map.fsize, MS_SYNC); /* Force write */
            lastup = time(0);
        }
    }
    free(tmp_comm_array);

    /*
    TODO : Move this to another function outside of timing step. this is collection of data post everything */
    double *x_loc, *y_loc, *fx, *fy, *x_vel, *y_vel;
    if (process_id != 0) {
        double *big_buff = malloc(count_per_process[process_id]*6*sizeof(double));
        x_loc = big_buff;
        y_loc = &big_buff[count_per_process[process_id]];
        fx = &big_buff[2*count_per_process[process_id]];
        fy = &big_buff[3*count_per_process[process_id]];
        x_vel = &big_buff[4*count_per_process[process_id]];
        y_vel = &big_buff[5*count_per_process[process_id]];

        //Accumulate data per process
        for (i=0; i<count_per_process[process_id]; i++) {
            int idx = particle_distribution[i];
            x_loc[i] = X(world, idx);
            y_loc[i] = Y(world, idx);
            fx[i] = XF(world, idx);
            fy[i] = YF(world, idx);
            x_vel[i] = XV(world, idx);
            y_vel[i] = YV(world, idx);
        }
        int tag = 0;
        MPI_Send(big_buff, 6*count_per_process[process_id], MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

        free(big_buff);
    } else {
        int **distribution = malloc (num_processes * sizeof(int*));
        for (i=0; i<num_processes; i++) {
            distribution[i] = malloc (count_per_process[i] * sizeof(int));
        }
        for (i = 0 ; i < world->bodyCt; i++) {
            int rem = i % num_processes;
            int div = i / num_processes;
            distribution[rem][div] = i;
        }
        double *buf;
        for (i=1; i<num_processes; i++) {
            buf = malloc(count_per_process[i]*6*sizeof(double));
            int tag = 0;
            MPI_Recv (buf, count_per_process[i]*6, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status );
            x_loc = buf;
            y_loc = &buf[count_per_process[i]];
            fx = &buf[2*count_per_process[i]];
            fy = &buf[3*count_per_process[i]];
            x_vel = &buf[4*count_per_process[i]];
            y_vel = &buf[5*count_per_process[i]];

            //Save the new values in the world struct
            for (j=0; j<count_per_process[i]; j++) {
                int particle = distribution[i][j]; //index of the particle whose values we are saving
                X(world, particle) = x_loc[j];
                Y(world, particle) = y_loc[j];
                XF(world, particle) = fx[j];
                YF(world, particle) = fy[j];
                XV(world, particle) = x_vel[j];
                YV(world, particle) = y_vel[j];
            }
            free(buf);
        }
        for(i=0; i<num_processes; i++) {
            free(distribution[i]);
        }
        free(distribution);
    }


}

void send_receive_initial_data(struct world *world, int process_id, int num_processes) 
{
    int world_size = world->bodyCt;
    MPI_Bcast (world->bodies.mass, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (world->bodies.radius, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (world->bodies.x1, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (world->bodies.y1, world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    world->bodies.x_old = world->bodies.x1;
    world->bodies.y_old = world->bodies.y1;
    world->bodies.x_new = world->bodies.x2;
    world->bodies.y_new = world->bodies.y2;

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

    unsigned int secsup;
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
    struct filemap image_map;
    struct world *world;
    int i;
    int bodies_per_process = 0;
    int *count_per_process;

    world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    if (argc != 5) {
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
    secsup = atoi(argv[2]);
    if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }
    steps = atoi(argv[4]);

    if (process_id == 0) {
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
    }

    bodies_per_process = (int) (world->bodyCt/num_processes);
    count_per_process = malloc(num_processes * sizeof(int));

    for (i=0; i<num_processes; i++) {
        if (i < world->bodyCt%num_processes) {
            count_per_process[i] = bodies_per_process + 1;
        } else {
            count_per_process[i] = bodies_per_process;
        }
    }

    send_receive_initial_data(world, process_id ,num_processes);

    if (process_id == 0) {
        if (gettimeofday(&start, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
    }

    /* Main Execution */
    do_compute(secsup, image_map, steps, world, process_id, num_processes, count_per_process);
    
    if (process_id == 0) {
        if (gettimeofday(&end, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
        rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) - 
                    (start.tv_sec + (start.tv_usec / 1000000.0));

        fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
        sleep(5);
        print(world);

        filemap_close(&image_map);
    }


    free(world);
    MPI_Finalize();

    return 0;
}
