#include "common.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include <ctype.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>


#include "vmath.h"		/* vector math macros */
#include "bu/app.h"
#include "bu/list.h"
#include "bu/time.h"
#include "bu/parse.h"
#include "bu/vls.h"
#include "bu/log.h"
#include "raytrace.h"		/* librt interface definitions */

#include "./rtuif.h"
#include "../../include/analyze/worker.h"
#include "analyze.h"
#include "../libanalyze/analyze_private.h"
#include <Python.h>
#include "NeuralRayTracer.h"
#include "ModelTrainer.h"
#include <fstream>
#include <iostream>
#include "../../include/bn/mat.h"
#include "./ext.h"
#include <cmath>
#include <vector>



#include "bio.h"
#include "bu/endian.h"
#include "bu/getopt.h"
#include "bu/bitv.h"
#include "bu/debug.h"
#include "bu/malloc.h"
#include "bu/parallel.h"
#include "bu/ptbl.h"

#include "bu/version.h"


#include "dm.h"
#include "pkg.h"


#include <iostream>
#include <cmath>
#include <random>
#include "brlcad_ident.h"

const char * progname;
const char * glob_database_name;
const char * object_name;

void
usage(const char *s)
{
    if (s) (void)fputs(s, stderr);
    bu_exit(1, "Usage: %s geom.g obj [obj...] < rayfile \n", progname);
}

/* 
Function to generate a random point on the surface of a sphere. 
This is used for the only external bounding box ray approach (Training Method #2). 
*/
std::pair<fastf_t, std::pair<fastf_t, fastf_t>> generateRandomPointOnSphere(
    point_t center, fastf_t radius) {

    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<fastf_t> uniformDist(0.0, 1.0);

    // Generate random spherical coordinates
    fastf_t theta = 2.0 * M_PI * uniformDist(gen);
    fastf_t phi = acos(2.0 * uniformDist(gen) - 1.0);

    // Convert spherical coordinates to Cartesian coordinates
    fastf_t x = center[0] + radius * sin(phi) * cos(theta);
    fastf_t y = center[1] + radius * sin(phi) * sin(theta);
    fastf_t z = center[2] + radius * cos(phi);

    // Return the random point
    return std::make_pair(x, std::make_pair(y, z));
}
/*
This function is used to write a point to the file for model training data. 
*/
char* pointToString(const point_t point, char* buffer, size_t bufferSize) {
    // Check for null pointer or insufficient buffer size
    if (point == NULL || buffer == NULL || bufferSize < 16) {
        return NULL;
    }

    // Initialize the string to an empty string
    buffer[0] = '\0';

    // Convert and concatenate each double to the string
    char temp[16];
    for (int i = 0; i < ELEMENTS_PER_POINT; i++) {
        snprintf(temp, sizeof(temp), "%.6f", point[i]);
        strcat(buffer, temp);

        // Add a comma and space after each element except the last one
        if (i < ELEMENTS_PER_POINT - 1) {
            strcat(buffer, ",");
        }

    }

    return buffer;
}

/**
 * rt_shootray() was told to call this on a hit.
 */
int
hit(struct application *ap, struct partition *PartHeadp, struct seg *UNUSED(segs))
{

    /* will serve as a pointer for the entry and exit hitpoints */
    struct hit *hitp;

    /* will serve as a pointer to the solid primitive we hit */
    struct soltab *stp;

    /* will contain surface curvature information at the entry */
    struct curvature cur = RT_CURVATURE_INIT_ZERO;

    /* will contain our hit point coordinate */
    point_t pt;

    /* will contain normal vector where ray enters geometry */
    vect_t inormal;

    /* will contain normal vector where ray exits geometry */
    vect_t onormal;

    /* Convert the origin to a string */
    char origin_str[64];
    pointToString(ap->a_ray.r_pt, origin_str, sizeof(origin_str));

    // convert direction to azimuth and elevation
    fastf_t azp;
    fastf_t elp;

    bn_ae_vec(&azp, &elp, ap->a_ray.r_dir);


    char az_el_str[100]; // Assuming the string will not exceed this length


    // Normalize az el
    azp = azp / 36.0;
    elp = elp / 9.0;

    // Store in az el string for training data
    sprintf(az_el_str, "%f,%f", azp, elp);


    /* First partition */
    struct partition *pp = PartHeadp->pt_forw;
    
    if (pp != PartHeadp) {  /* Ensure we have a partition */

        /* entry hit point */
        hitp = pp->pt_inhit;

        /* construct the actual (entry) hit-point from the ray */
        VJOIN1(pt, ap->a_ray.r_pt, hitp->hit_dist, ap->a_ray.r_dir);

        // Store hitpoint information
        fastf_t hitpoint_x = pt[0];
        fastf_t hitpoint_y = pt[1];
        fastf_t hitpoint_z = pt[2];

        // Convert hitpoint information to string
        char hitpoint_str[100];
        sprintf(hitpoint_str, "%f,%f,%f", hitpoint_x, hitpoint_y, hitpoint_z);

        // Store distance from ray origin to hit point
        fastf_t distance = hitp->hit_dist;

        // Write to distance string for training data
        char distance_str[100]; // Assuming the string will not exceed this length
        sprintf(distance_str, "%f", distance);

        /* primitive we encountered on entry */
        stp = pp->pt_inseg->seg_stp;

        /* compute the normal vector at the entry point */
        RT_HIT_NORMAL(inormal, hitp, stp, &(ap->a_ray), pp->pt_inflip);

        // convert normal vector direction to azimuth and elevation
        fastf_t azp2;
        fastf_t elp2;

        bn_ae_vec(&azp2, &elp2, inormal);


        char az_el_normal_str[100]; // Assuming the string will not exceed this length

        // Normalize Az El
        azp2 = azp2 / 36.0;
        elp2 = elp2 / 9.0;

        // Write normal vector az el to file for training data
        sprintf(az_el_normal_str, "%f,%f", azp2, elp2);

        /* Open file, write, and close */
        const char* hit_or_miss_file_name = "hit_or_miss_file.txt";
        FILE* hit_or_miss_file = fopen(hit_or_miss_file_name, "a");
        if (hit_or_miss_file == NULL) {
            perror("Error opening file");
            exit(0);  // Exit here is rather harsh, consider returning with an error code instead
        }

        /* Open file, write, and close */

        /* Open file, write, and close */

        /*

        ---CODE FOR TRAINING METHOD 2


        const char* bounding_sphere_rays = "bounding_sphere_rays_file.txt";
        FILE* bounding_sphere_rays_file = fopen(bounding_sphere_rays, "a");
        if (bounding_sphere_rays_file == NULL) {
            perror("Error opening file");
            exit(0);  // Exit here is rather harsh, consider returning with an error code instead
        }
        */



        /*

        CODE FOR TRAINING SHADING MODEL

        const char* hit_file_name = "hit_file.txt";
        FILE* hit_file = fopen(hit_file_name, "a");
        if (hit_file == NULL) {
            perror("Error opening file");
            exit(0);  // Exit here is rather harsh, consider returning with an error code instead
        }
        */

        const char* hit_str = "1";

        // Perform write to training files. 
        for(int w = 0; w < 5; w++) {
            fprintf(hit_or_miss_file, "%s*", origin_str);
            fprintf(hit_or_miss_file, "%s*", az_el_str);
            fprintf(hit_or_miss_file, "%s\n", hit_str);

            /*

            --- CODE FOR TRAINING METHOD 2


            fprintf(bounding_sphere_rays_file, "%s*", origin_str);
            fprintf(bounding_sphere_rays_file, "%s*", az_el_str);
            fprintf(bounding_sphere_rays_file, "%s*", hit_str);
            fprintf(bounding_sphere_rays_file, "%s\n", distance_str);
            */


            /*
            
            CODE FOR TRAINING SHADING MODEL

            if(w == 0) {
                fprintf(hit_file, "%s*", origin_str);
                fprintf(hit_file, "%s*", az_el_str);
                fprintf(hit_file, "%s*", distance_str);
                fprintf(hit_file, "%s\n", az_el_normal_str);

            }
            */
        }
        
        fclose(hit_or_miss_file);
        // fclose(bounding_sphere_rays_file); CODE FOR TRAINING METHOD 2
        // fclose(hit_file); // For training shading model
    }


    /* A more complicated application would probably fill in a new
     * local application structure and describe, for example, a
     * reflected or refracted ray, and then call rt_shootray() for
     * those rays.
     */

    /* Hit routine callbacks generally return 1 on hit or 0 on miss.
     * This value is returned by rt_shootray().
     */
    return 1;
}


/**
 * This is a callback routine that is invoked for every ray that
 * entirely misses hitting any geometry.  This function is invoked by
 * rt_shootray() if the ray encounters nothing.
 */
int
miss(struct application *ap)
{

    // Print to file
    const char* hit_or_miss_file_name = "hit_or_miss_file.txt";

    // Open the file in append mode (creates if it doesn't exist)
    FILE* hit_or_miss_file = fopen(hit_or_miss_file_name, "a");

    if (hit_or_miss_file == NULL) {
        // Handle error if unable to open the file
        perror("Error opening hit_or_miss_file");
        exit(0); // may cause errors?
    }


    /*

     ---CODE FOR TRAINING METHOD 2


    // Print to file
    const char* bounding_sphere_rays = "bounding_sphere_rays_file.txt";

    // Open the file in append mode (creates if it doesn't exist)
    FILE* bounding_sphere_rays_file = fopen(bounding_sphere_rays, "a");

    if (bounding_sphere_rays == NULL) {
        // Handle error if unable to open the file
        perror("Error opening hit_or_miss_file");
        exit(0); // may cause errors?
    }
    */

    /* Convert the origin to a string */
    char origin_str[64];
    pointToString(ap->a_ray.r_pt, origin_str, sizeof(origin_str));

    // convert direction to azimuth and elevation
    fastf_t azp;
    fastf_t elp;

    bn_ae_vec(&azp, &elp, ap->a_ray.r_dir);

    // Normalize az el
    azp = azp / 36.0;
    elp = elp / 9.0;

    char az_el_str[100]; // Assuming the string will not exceed this length

    const char* miss_str = "0";
    const char * miss_dist = "0.0";

    // Store az el string for ray direction
    sprintf(az_el_str, "%f,%f", azp, elp);
    // Write ray origin
    fprintf(hit_or_miss_file, "%s*", origin_str);
    // Write ray direction
    fprintf(hit_or_miss_file, "%s*", az_el_str);

    // Write 0 for miss
    fprintf(hit_or_miss_file, "%s\n", miss_str);

    /*

     ---CODE FOR TRAINING METHOD 2

    // Write ray origin
    fprintf(bounding_sphere_rays_file, "%s*", origin_str);
    // Write ray direction
    fprintf(bounding_sphere_rays_file, "%s*", az_el_str);

    // Write 0 for miss
    fprintf(bounding_sphere_rays_file, "%s*", miss_str);

    // Write 0 for miss
    fprintf(bounding_sphere_rays_file, "%s\n", miss_dist);

    fclose(bounding_sphere_rays_file);
    */
    
    // Close the file when done
    fclose(hit_or_miss_file);
    
    return 0;
}

static int
op_overlap(struct application *ap, struct partition *UNUSED(pp),
		struct region *UNUSED(reg1), struct region *UNUSED(reg2),
		struct partition *UNUSED(hp))
{
    RT_CK_APPLICATION(ap);
    return 0;
}

/*
Used by Training Method #1 (bounding box with internal rays) to get random rays. 
*/
void
get_random_rays(fastf_t *rays, long int craynum, point_t center, fastf_t radius)
{

    long int i = 0;
    point_t p1, p2;

    fastf_t radius_proxy = radius;

    if (!rays) return;
    for (i = 0; i < craynum; i++) {

        if(i < craynum / 5) {
            radius_proxy = radius;
        } else if (i > craynum / 5 && i < 2 * craynum / 5) {
            radius_proxy = radius * 2;
        } else if (i > 2 * craynum / 5 && i < 3 * craynum / 5) {
            radius_proxy = radius * 3;
        } else if (i > 3 * craynum / 5 && i < 4 * craynum / 5) {
            radius_proxy = radius * 4;
        } else {
            radius_proxy = radius * 5;
        }

    
	vect_t n;
	bn_rand_sph_sample(p1, center, radius_proxy);
	bn_rand_sph_sample(p2, center, radius_proxy);
	VSUB2(n, p2, p1);
	VUNITIZE(n);
	rays[i*6+0] = p1[X];
	rays[i*6+1] = p1[Y];
	rays[i*6+2] = p1[Z];
	rays[i*6+3] = n[X];
	rays[i*6+4] = n[Y];
	rays[i*6+5] = n[Z];
    }
}

/*
Used by training method #3 to get a grid to the right of the geometry
(along +Y axis)
*/
void get_right_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0];
    p1[2] = center[2];
    p1[1] = center[1] + radius;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix Y position
    fastf_t y = center[1] + (radius * radius_factor);

    // Manipulate X and Z to form a grid
    for (fastf_t z = center[2] - (radius * 1.2); z < center[2] + (radius * 1.2); z += 0.2) {
        for (fastf_t x = center[0] - (radius * 1.2); x < center[0] + (radius * 1.2); x += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }
}

/*
Used by training method #3 to get a grid to the left of the geometry
(along -Y axis)
*/
void get_left_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0];
    p1[2] = center[2];
    p1[1] = center[1] - radius;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix Y position
    fastf_t y = center[1] - (radius * radius_factor);

    // Manipulate X and Z to form a grid
    for (fastf_t z = center[2] - (radius * 1.2); z < center[2] + (radius * 1.2); z += 0.2) {
        for (fastf_t x = center[0] - (radius * 1.2); x < center[0] + (radius * 1.2); x += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}

/*
Used by training method #3 to get a grid in front of the geometry
(along +X axis)
*/
void get_front_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {
    
    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0] + radius;
    p1[1] = center[1];
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix X position
    fastf_t x = center[0] + (radius * radius_factor);

    // Manipulate Y and Z to form a grid
    for (fastf_t z = center[2] - (radius * 1.2); z < center[2] + (radius * 1.2); z += 0.2) {
        for (fastf_t y = center[1] - (radius * 1.2); y < center[1] + (radius * 1.2); y += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}

/*
Used by training method #3 to get a grid above the geometry
(along +Z axis)
*/
void get_top_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0];
    p1[1] = center[1];
    p1[2] = center[2] + radius;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix Z position
    fastf_t z = center[2] + (radius * radius_factor);

    // Manipulate X and Y to form a grid
    for (fastf_t x = center[0] - (radius * 1.2); x < center[0] + (radius * 1.2); x += 0.2) {
        for (fastf_t y = center[1] - (radius * 1.2); y < center[1] + (radius * 1.2); y += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}

/*
Used by training method #3 to get a grid below the geometry
(along -Z axis)
*/
void get_bot_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0];
    p1[1] = center[1];
    p1[2] = center[2] - radius;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix Z position
    fastf_t z = center[2] - (radius * radius_factor);

    // Manipulate X and Y to form a grid
    for (fastf_t x = center[0] - (radius * 1.2); x < center[0] + (radius * 1.2); x += 0.2) {
        for (fastf_t y = center[1] - (radius * 1.2); y < center[1] + (radius * 1.2); y += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}

/*
Used by training method #3 to get a behind  the geometry
(along -X axis)
*/
void get_back_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    /* Find the direction vector from grid to center*/
    point_t p1;
    p1[0] = center[0] - radius;
    p1[1] = center[1];
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    // Fix X position
    fastf_t x = center[0] - (radius * radius_factor);

    // Manipulate Y and Z to form a grid
    for (fastf_t y = center[1] - (radius * 1.2); y < center[1] + (radius * 1.2); y += 0.2) {
        for (fastf_t z = center[2] - (radius * 1.2); z < center[2] + (radius * 1.2); z += 0.2) {

            std::vector<fastf_t> current_ray;

            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }
}



/*
This is used to get a grid between X and Y axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis and it varies in the -Z to +Z direction.  
*/
void get_mid_one_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;
    fastf_t new_x = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_x;
    p1[0] = center[0] + new_x;
    p1[1] = center[1] + new_y;
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);


    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t x1 = 0;
    fastf_t y1 = intersect;
    fastf_t end_x = intersect;
    fastf_t end_y = 0;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t x = x1 + t * (end_x - x1);
        fastf_t y = y1 + t * (end_y - y1);

        for(fastf_t z = center[2] - (radius * 1.4); z < center[2] + (radius * 2); z += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}


/*
This is used to get a grid between -X and Y axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis and it varies in the -Z to +Z direction.  
*/
void get_mid_two_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_x = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_x;

    p1[0] = center[0] - new_x;
    p1[1] = center[1] + new_y;
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);


    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t x1 = 0;
    fastf_t y1 = intersect;
    fastf_t end_x = intersect * -1;
    fastf_t end_y = 0;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t x = x1 + t * (end_x - x1);
        fastf_t y = y1 + t * (end_y - y1);

        for(fastf_t z = center[2] - (radius * 1.4); z < center[2] + (radius * 2); z += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}

/*
This is used to get a grid between -X and -Y axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis and it varies in the -Z to +Z direction.  
*/
void get_mid_three_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_x = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_x;

    p1[0] = center[0] - new_x;
    p1[1] = center[1] - new_y;
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t x1 = 0;
    fastf_t y1 = intersect * -1;
    fastf_t end_x = intersect * -1;
    fastf_t end_y = 0;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t x = x1 + t * (end_x - x1);
        fastf_t y = y1 + t * (end_y - y1);

        for(fastf_t z = center[2] - (radius * 1.4); z < center[2] + (radius * 2); z += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}


/*
This is used to get a grid between X and -Y axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis and it varies in the -Z to +Z direction.  
*/
void get_mid_four_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_x = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_x;

    p1[0] = center[0] + new_x;
    p1[1] = center[1] - new_y;
    p1[2] = center[2];
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t x1 = 0;
    fastf_t y1 = intersect * -1;
    fastf_t end_x = intersect ;
    fastf_t end_y = 0;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t x = x1 + t * (end_x - x1);
        fastf_t y = y1 + t * (end_y - y1);

        for(fastf_t z = center[2] - (radius * 1.4); z < center[2] + (radius * 2); z += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }
}

/*
This is used to get a grid between -Y and Z axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis.
*/


void get_slant_one_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_z = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_z;

    p1[0] = center[0];
    p1[1] = center[1] - new_y;
    p1[2] = center[2] + new_z;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t y1 = -1 * intersect;
    fastf_t z1 = 0;
    fastf_t end_y = 0;
    fastf_t end_z = intersect;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t y = y1 + t * (end_y - y1);
        fastf_t z = z1 + t * (end_z - z1);
        

        for(fastf_t x = center[0] - (radius * 1.4); x < center[0] + (radius * 1.4); x += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}


/*
This is used to get a grid between Y and Z axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis.
*/
void get_slant_two_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_z = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_z;

    p1[0] = center[0];
    p1[1] = center[1] + new_y;
    p1[2] = center[2] + new_z;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);


    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t y1 = intersect;
    fastf_t z1 = 0;
    fastf_t end_y = 0;
    fastf_t end_z = intersect;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t y = y1 + t * (end_y - y1);
        fastf_t z = z1 + t * (end_z - z1);
        

        for(fastf_t x = center[0] - (radius * 1.4); x < center[0] + (radius * 1.4); x += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}


/*
This is used to get a grid between Y and -Z axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis.
*/
void get_slant_three_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_z = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_z;

    p1[0] = center[0];
    p1[1] = center[1] + new_y;
    p1[2] = center[2] - new_z;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t y1 = intersect;
    fastf_t z1 = 0;
    fastf_t end_y = 0;
    fastf_t end_z =  -1 * intersect;

    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t y = y1 + t * (end_y - y1);
        fastf_t z = z1 + t * (end_z - z1);

        

        for(fastf_t x = center[0] - (radius * 1.4); x < center[0] + (radius * 1.4); x += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }

    }

}


/*
This is used to get a grid between -Y and -Z axis. This is used
if attempting to do the grid approach from multiple angles. The edges of the grid touch both axis.
*/
void get_slant_four_grid(std::vector<std::vector<fastf_t>> &current_grid, point_t center, fastf_t radius, fastf_t radius_factor) {

    // get direction first
    point_t p1;
    fastf_t radius_proxy = radius * radius_factor;

    fastf_t new_z = sqrt(radius_proxy * radius_proxy / 2);
    fastf_t new_y = new_z;

    p1[0] = center[0];
    p1[1] = center[1] - new_y;
    p1[2] = center[2] - new_z;
    vect_t n;
    VSUB2(n, center, p1);
    VUNITIZE(n);

    fastf_t intersect = radius_proxy / sqrt(2);

    // Starting point is (intersect, 0) and end is (0, intersect)
    fastf_t y1 = -1 * intersect;
    fastf_t z1 = 0;
    fastf_t end_y = 0;
    fastf_t end_z =  -1 * intersect;


    for (fastf_t t = 0.0; t <= 1.0; t += 0.01) {
        fastf_t y = y1 + t * (end_y - y1);
        fastf_t z = z1 + t * (end_z - z1);

        for(fastf_t x = center[0] - (radius * 1.4); x < center[0] + (radius * 1.4); x += 0.1) {
            std::vector<fastf_t> current_ray;
            current_ray.push_back(x);
            current_ray.push_back(y);
            current_ray.push_back(z);
            current_ray.push_back(n[X]);
            current_ray.push_back(n[Y]);
            current_ray.push_back(n[Z]);

            current_grid.push_back(current_ray);

        }
    }

}

/*
This function is used for the training method which shoots rays from
a bounding sphere with internal rays. (Training Method #1)

*/
void
analyze_prand_pnt_worker(int cpu, void *ptr)
{
    struct application ap;
    struct rt_gen_worker_vars *state = &(((struct rt_gen_worker_vars *)ptr)[cpu]);
    size_t i;

    RT_APPLICATION_INIT(&ap);
    ap.a_rt_i = state->rtip;
    ap.a_hit = state->fhit;
    ap.a_miss = state->fmiss;
    ap.a_overlap = state->foverlap;
    ap.a_onehit = 1;
    ap.a_logoverlap = rt_silent_logoverlap;
    ap.a_resource = state->resp;
    ap.a_uptr = (void *)state;

    for (i = 0; i < state->rays_cnt; i++) {
	VSET(ap.a_ray.r_pt, state->rays[6*i+0], state->rays[6*i+1], state->rays[6*i+2]);
	VSET(ap.a_ray.r_dir, state->rays[6*i+3], state->rays[6*i+4], state->rays[6*i+5]);
	rt_shootray(&ap);
    }
}


/*
This is used for gathering training data through the form of grid rays (Training Method #3). 
This performs the actual firing of the rays. 

*/
void
shoot_grid_rays(std::vector<std::vector<std::vector<fastf_t>>> grids, struct rt_i *rtip)
{

    struct application ap;

    RT_APPLICATION_INIT(&ap);
    ap.a_rt_i = rtip;
    ap.a_hit = hit;
    ap.a_miss = miss;
    ap.a_onehit = 1;
    ap.a_logoverlap = rt_silent_logoverlap;

    for(int i = 0; i < grids.size(); i++) {
        std::vector<std::vector<fastf_t>> current_grid = grids[i];

        for(int j = 0; j < current_grid.size(); j++) {
            std::vector<fastf_t> current_ray = current_grid[j];
            VSET(ap.a_ray.r_pt, current_ray[0], current_ray[1], current_ray[2]);
	        VSET(ap.a_ray.r_dir, current_ray[3], current_ray[4], current_ray[5]);
            rt_shootray(&ap);

        }
    }
}

/* 
This is used for the only external bounding box ray approach (Training Method #2). 
It takes a point on the bounding sphere and creates a direction vector
to the center of the geometry, and fires it.
*/
void shoot_random_point(const std::pair<fastf_t, std::pair<fastf_t, fastf_t>>& randomPoint, struct rt_i *rtip, point_t center_point) {
    fastf_t x = randomPoint.first;
    fastf_t y = randomPoint.second.first;
    fastf_t z = randomPoint.second.second;

    struct application ap;

    RT_APPLICATION_INIT(&ap);
    ap.a_rt_i = rtip;
    ap.a_hit = hit;
    ap.a_miss = miss;
    ap.a_onehit = 1;
    ap.a_logoverlap = rt_silent_logoverlap;

    point_t p1;
    p1[X] = x;
    p1[Y] = y;
    p1[Z] = z;

    point_t p2;

    // Generate and print a random point
    std::pair<fastf_t, std::pair<fastf_t, fastf_t>> random_second_point = generateRandomPointOnSphere(center_point, rtip->rti_radius);

    fastf_t x2 = random_second_point.first;
    fastf_t y2 = random_second_point.second.first;
    fastf_t z2 = random_second_point.second.second;

    p2[X] = x2;
    p2[Y] = y2;
    p2[Z] = z2;

    vect_t n;
    VSUB2(n, p2, p1);
    VUNITIZE(n);

    VSET(ap.a_ray.r_pt, x, y, z);
    VSET(ap.a_ray.r_dir, n[X], n[Y], n[Z]);
    rt_shootray(&ap);
}



int main(int argc, char **argv) {

    /* every application needs one of these */
    struct application	ap;
    static struct rt_i *rtip;	/* rt_dirbuild returns this */
    char idbuf[2048] = {0};	/* First ID record info */

    int status = 0;
    struct bu_vls buf = BU_VLS_INIT_ZERO;
    //struct shot sh;

    bu_setprogname(argv[0]);
    progname = argv[0];
    glob_database_name = argv[1];
    object_name = argv[2];

    if (argc < 3) {
	usage("insufficient args\n");
    }

    /* initialize the application structure to all zeros */
    RT_APPLICATION_INIT(&ap);

    /*
     *  Load database.
     *  rt_dirbuild() returns an "instance" pointer which describes
     *  the database to be ray traced.  It also gives you back the
     *  title string in the header (ID) record.
     */
    
    
    if ( (rtip=rt_dirbuild(glob_database_name, idbuf, sizeof(idbuf))) == RTI_NULL ) {
	bu_exit(2, "Error building dir in train neural!\n");
    }
    

    ap.a_rt_i = rtip;	/* your application uses this instance */

    /* Walk trees.
     * Here you identify any object trees in the database that you
     * want included in the ray trace.
     */

    while (argc > 2) {
	if ( rt_gettree(rtip, argv[2]) < 0 )
	   fprintf(stderr, "rt_gettree(%s) FAILED\n", argv[0]);
	argc--;
	argv++;
    }

    // Convert the C-style string to a C++ string
    std::string temp__global_db_name(glob_database_name);

     // Create a new variable with everything except the last two characters (.g)
    std::string temp_db_name = temp__global_db_name.substr(0, temp__global_db_name.size() - 2);

     // Check if file exists in "./model_weights/"
    std::string file_path = std::string("./model_weights/") + temp_db_name + "_" + object_name + "_weights.pth";
    std::ifstream file(file_path);

    // If model weights already exist, no need to fire training rays again (model will not be retrained). 
    if(file) {
        std::cout << "Model 1 has already been trained. Use existing model weights" << std::endl; 
        return 0;
    }


    
    int pntcnt = 0;
    size_t i;
    int ret, j;
    int do_grid = 1;
    fastf_t oldtime, currtime;
    int ind = 0;
    int count = 0;
    double avgt = 0.0;
    size_t ncpus = bu_avail_cpus();
    struct rt_gen_worker_vars *state = (struct rt_gen_worker_vars *)bu_calloc(ncpus+1, sizeof(struct rt_gen_worker_vars ), "state");
    struct bu_ptbl **grid_pnts = NULL;
    struct bu_ptbl **rand_pnts = NULL;
    struct bu_ptbl **sobol_pnts = NULL;
    struct resource *resp = (struct resource *)bu_calloc(ncpus+1, sizeof(struct resource), "resources");
    int pntcnt_grid = 0;
    int pntcnt_rand = 0;
    int pntcnt_sobol = 0;

    oldtime = bu_gettime();
    
    for (i = 0; i < ncpus+1; i++) {
        /* standard */
        state[i].rtip = rtip;
        state[i].fhit = hit;
        state[i].fmiss = miss;
        state[i].foverlap = op_overlap;
        state[i].resp = &resp[i];
        state[i].ind_src = &ind;
        rt_init_resource(state[i].resp, (int)i, rtip);
    }

    while (argc > 2)  { 
    if (rt_gettree(rtip, argv[2]) < 0)
       bu_log("Loading the geometry for [%s] FAILED\n", argv[2]);
    argc--;
    argv++;
    }

    rt_prep_parallel(rtip, (int)ncpus);


    currtime = bu_gettime();

    /* stop at the first point of intersection or shoot all the way
     * through (defaults to 0 to shoot all the way through).
     */
    ap.a_onehit = 1;

    /* This is what callback to perform on a hit. */
    ap.a_hit = hit;

    /* This is what callback to perform on a miss. */
    ap.a_miss = miss;

    /*

    --- THIS CHUNK OF COMMENTED OUT CODE SHOOTS TRAINING RAYS IN THE FORM OF A BOUNDING BOX. 
    --- HOWEVER, THERE ARE INTERNAL RAYS FIRED FROM WITHIN THIS BOUNDING BOX AS WELL. 
    --- Training Method #1

    fastf_t *rays;
	grid_pnts = (struct bu_ptbl **)bu_calloc(ncpus+1, sizeof(struct bu_ptbl *), "local state");
	
    for (i = 0; i < ncpus+1; i++) {

	   BU_GET(grid_pnts[i], struct bu_ptbl);
	   bu_ptbl_init(grid_pnts[i], 64, "first and last hit points");
	   state[i].ptr = (void *)grid_pnts[i];
	}


    struct bn_tol btol = BN_TOL_INIT_TOL;
    point_t rpp_min, rpp_max;
    btol.dist = DIST_PNT_PNT(rpp_max, rpp_min) * 0.01;

    count = analyze_get_bbox_rays(&rays, rtip->mdl_min, rtip->mdl_max, &btol);

	for (i = 0; i < ncpus+1; i++) {
	   state[i].step = (int)(count/ncpus * 0.1);
	   state[i].rays_cnt = count;
	   state[i].rays = rays;
	}

    oldtime = bu_gettime();
	//bu_parallel(analyze_gen_worker, ncpus, (void *)state);
	currtime = bu_gettime();

    
	for (i = 0; i < ncpus+1; i++) {
	   state[i].rays = NULL;
	}

    int max_pnts = 1000;

    size_t mrc = 1000000;
    
    size_t ccnt = (ncpus >= LONG_MAX-1) ? ncpus : ncpus+1;
    size_t craynum = mrc/ccnt;
	fastf_t mt = (fastf_t)INT_MAX;

    point_t center;

    VADD2SCALE(center, rtip->mdl_max, rtip->mdl_min, 0.5);

    
    // Random
    fastf_t delta = 0;
    size_t raycnt = 0;
    int pc = 0;
    rand_pnts = (struct bu_ptbl **)bu_calloc(ncpus+1, sizeof(struct bu_ptbl *), "local state");
    
    for (i = 0; i < ncpus+1; i++) {

       BU_GET(rand_pnts[i], struct bu_ptbl);
       bu_ptbl_init(rand_pnts[i], 64, "first and last hit points");
       state[i].ptr = (void *)rand_pnts[i];
       if (!state[i].rays) {
           state[i].rays = (fastf_t *)bu_calloc(craynum * 6 + 1, sizeof(fastf_t), "rays");
           state[i].rays_cnt = craynum;
       }
    }

    
    oldtime = bu_gettime();
    while (delta < mt && raycnt < mrc && (!max_pnts || pc < max_pnts)) {
        for (i = 0; i < ncpus+1; i++) {
            get_random_rays(state[i].rays, craynum, center, rtip->rti_radius);
        }
        bu_parallel(analyze_prand_pnt_worker, ncpus, (void *)state);
        raycnt += craynum * (ncpus+1);
        pc = 0;
        for (i = 0; i < ncpus+1; i++) {
            pc += (int)BU_PTBL_LEN(rand_pnts[i]);
        }
        if (max_pnts && (pc >= max_pnts)) break;
        delta = (bu_gettime() - oldtime)/1e6;
    }
    */
    

    /*

    --- THIS CHUNK OF COMMENTED OUT CODE SHOOTS TRAINING RAYS IN THE FORM OF A BOUNDING BOX.
    --- IT FINDS RANDOM POINTS ON THE BOUNDING SPHERE AND USES THIS AS THE RAY ORIGIN. TO
    --- GET THE DIRECTION VECTOR FOR THE RAY, IT GETS THE DIRECTION FROM THE POINT ON
    --- THE BOUNDING BOX TO THE CENTER POINT OF THE GEOMETRY. 
    --- Training Method #2
    

    point_t center;

    VADD2SCALE(center, rtip->mdl_max, rtip->mdl_min, 0.5);

    int num_rays_to_fire = 1000000;

    // Generate a vector of random points
    std::vector<std::pair<fastf_t, std::pair<fastf_t, fastf_t>>> randomPoints;

    for(int i = 0; i < num_rays_to_fire; i++) {
        // Generate and print a random point
        std::pair<fastf_t, std::pair<fastf_t, fastf_t>> randomPoint = generateRandomPointOnSphere(center, rtip->rti_radius);

        randomPoints.push_back(randomPoint);
    }

    for(int i = 0; i < randomPoints.size(); i++) {
        shoot_random_point(randomPoints[i], rtip, center);
    }
    */
    

    /*
    --- THE BELOW CHUNK OF CODE SHOOTS TRAINING RAYS IN THE FORM OF GRIDS. 
    --- THIS IS SHOOTING FROM THE PERSPECTIVE OF AZ = 90 EL = 0
    --- THESE ARE STORED IN THE "right_grids" VECTOR. THERE ARE 
    --- SIMILAR FUNCTIONS TO SHOOT GRIDS FROM VARIOUS PERSPECTIVES (LEFT, TOP, BOT, MID_ONE, ETC)
    --- Training Method #3
    */

    

    point_t center;

    VADD2SCALE(center, rtip->mdl_max, rtip->mdl_min, 0.5);
  
    std::vector<std::vector<std::vector<fastf_t>>> right_grids;

    for(fastf_t factor = 1; factor <= 2; factor += 0.3) {
        std::vector<std::vector<fastf_t>> right_grid;
        get_right_grid(right_grid, center, rtip->rti_radius, factor);
        right_grids.push_back(right_grid);
    }

    shoot_grid_rays(right_grids, rtip);
    

    // Create object and call constructor of class to train the model on this geometry
    ModelTrainer trainer(glob_database_name, object_name);
    

    /*
    
    --- CODE FOR TRAINING SHADING MODEL

    const char* filename = "hit_file.txt";
    if (std::remove(filename) != 0) {
        std::cerr << "Error deleting hit_file.txt" << std::endl;
    }
    */

    // Delete the training ray files now that the model has been trained
    const char* filename2 = "hit_or_miss_file.txt";

    if (std::remove(filename2) != 0) {
        std::cerr << "Error deleting hit_or_miss_file.txt"  << std::endl;
    }
    
    return 0;

}