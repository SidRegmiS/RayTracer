//Sid Regmi
//ssregmi
//cs4611

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include "math.h"


//these are to define the image height and width and the channels
#define height (512)
#define width (512)
#define channels (3)


//the ray struct
typedef struct {
    float startPosition[3]; //starting position in the camera
    float directionVector[3]; //direction it is going
} Ray;


typedef struct {
    float color[3];
    int reflective;
} material;

typedef struct {
    float center[3]; //the center point of the sphere
    float radius; //radius of the sphere
    material mat;
} Sphere;

typedef struct {
    float time;
    material rayHitMat;
    float intersectLocation[3];
    float normal[3];
} RayHit;


typedef struct {
    float v[3][3];
    material mat;
} Triangle;


//math functions
void normalize(float *vector){
    float xSquare = vector[0] * vector[0];
    float ySquare = vector[1] * vector[1];
    float zSquare = vector[2] * vector[2];

    float adding = xSquare + ySquare + zSquare;

    float norm = sqrtf(adding);

    float normX = vector[0]/norm;
    float normY = vector[1]/norm;
    float normZ = vector[2]/norm;

    vector[0] = normX;
    vector[1] = normY;
    vector[2] = normZ;

}

//helper function for subtracting and doing dot product of vectors

void subtractVector( const float vector[3],  const float vector2[3], float vectorDifference[3]){
    vectorDifference[0] = vector[0] - vector2[0];
    vectorDifference[1] = vector[1] - vector2[1];
    vectorDifference[2] = vector[2] - vector2[2];
}

float dotProduct(const float vector[3], const float vector2[3]){
    float dot = 0;
    dot += (vector[0] * vector2[0]);
    dot += (vector[1] * vector2[1]);
    dot += (vector[2] * vector2[2]);

    return dot;
}

//this gets the ray from the screen position
void getRay(int x, int y, Ray *ray){


    float pixelPosition[3];
    pixelPosition[0] = (x-255.5)/256;
    pixelPosition[1] =  ((511-y)-255.5)/256;
    pixelPosition[2] = -2;

    float ray_Vector[3];

    //normalize the position to get the direction
    ray_Vector[0] = pixelPosition[0];
    ray_Vector[1] = pixelPosition[1];
    ray_Vector[2] = pixelPosition[2];

    //normalize
    normalize(ray_Vector);

    //set the ray.  will always be 0,0,0
    ray->startPosition[0] = 0;
    ray->startPosition[1] = 0;
    ray->startPosition[2] = 0;

    ray->directionVector[0] = ray_Vector[0];
    ray->directionVector[1] = ray_Vector[1];
    ray->directionVector[2] = ray_Vector[2];

}

void scaleMultiply(float scale, const float vector[3], float out[3]){
    out[0] = vector[0] * scale;
    out[1] = vector[1] * scale;
    out[2] = vector[2] * scale;
}

void addVectors(const float vector[3], const float vector2[3], float out[3]){
    out[0] = vector[0] + vector2[0];
    out[1] = vector[1] + vector2[1];
    out[2] = vector[2] + vector2[2];
}

void crossProduct(const float vector[3], const float vector2[3], float out[3]){
    float u1 = vector[0];
    float u2 = vector[1];
    float u3 = vector[2];

    float v1 = vector2[0];
    float v2 = vector2[1];
    float v3 = vector2[2];

    out[0] = (u2*v3) - (u3*v2);
    out[1] = (u3*v1) - (u1*v3);
    out[2] = (u1*v2) - (u2*v1);
}




//to get when a ray intersects with sphere. 
RayHit sphereIntersection(Ray *ray, Sphere *sphere){
    RayHit rayHit = {.time = -1, .intersectLocation = {0,0,0}, .normal = {0,0,0}};

    float e_Sub_c[3] ={0,0,0};
    subtractVector(ray->startPosition, sphere->center,e_Sub_c);
    float d_Dot_e_Sub_c = dotProduct(ray->directionVector, e_Sub_c);
    float d_Dot_e_Sub_c_Squared = d_Dot_e_Sub_c * d_Dot_e_Sub_c;

    float d_dot_d = dotProduct(ray->directionVector, ray->directionVector);
    float e_Sub_c_Dot_e_Sub_c = dotProduct(e_Sub_c, e_Sub_c);
    float radiusSquared = sphere->radius*sphere->radius;
    float e_Sub_c_Dot_e_Sub_c_Minus_radiusSquared = e_Sub_c_Dot_e_Sub_c - radiusSquared;

    float d_dot_d_MUL_e_Sub_c_Dot_e_Sub_c_Minus_radiusSquared = d_dot_d * e_Sub_c_Dot_e_Sub_c_Minus_radiusSquared;


    float discriminant = d_Dot_e_Sub_c_Squared - d_dot_d_MUL_e_Sub_c_Dot_e_Sub_c_Minus_radiusSquared;

    //if you miss
    if(discriminant < 0){
        rayHit.time = -1;
        return rayHit;
    }

    rayHit.rayHitMat = sphere->mat;

    //find the smallest t value that is positive
    //get -d
    float negD[3];
    float zeros[3] = {0, 0,0};

    subtractVector(zeros,ray->directionVector, negD);
    float negD_dot_e_Sub_c = dotProduct(negD,e_Sub_c);

    float time1 = (negD_dot_e_Sub_c + sqrt(discriminant));
    time1 = time1/d_dot_d;
    float time2 = (negD_dot_e_Sub_c - sqrt(discriminant));
    time2 = time2/d_dot_d;

    if(time1 > 0 && time1 < time2){
        rayHit.time = time1;
    }
    else if(time2 > 0 && time2 < time1) {
        rayHit.time = time2;
    }

    //store the intersection point
    float hitPositions[3];
    float dt[3];

    scaleMultiply(rayHit.time, ray->directionVector, dt);
    addVectors(ray->startPosition, dt, hitPositions);

    rayHit.intersectLocation[0] = hitPositions[0];
    rayHit.intersectLocation[1] = hitPositions[1];
    rayHit.intersectLocation[2] = hitPositions[2];

    //setting the normal
    float normal[3];
    subtractVector(rayHit.intersectLocation, sphere->center,normal);
    normalize(normal);

    rayHit.normal[0] = normal[0];
    rayHit.normal[1] = normal[1];
    rayHit.normal[2] = normal[2];




    return rayHit;
}

//for when a ray hits a triangle.
RayHit triangleIntersect(Ray *ray, Triangle *triangle){
    RayHit  rayHit = {.time = -1};

    //doing this based on the slides from lecture
    float A, B, C, D, E, F, G, H, I, J, K, L;

    float M, beta, gamma, t;

    int pointA = 0, x = 0;
    int pointB = 1, y = 1;
    int pointC = 2, z = 2;

    A = triangle->v[pointA][x] - triangle->v[pointB][x];
    B = triangle->v[pointA][y] - triangle->v[pointB][y];
    C = triangle->v[pointA][z] - triangle->v[pointB][z];

    D = triangle->v[pointA][x] - triangle->v[pointC][x];
    E = triangle->v[pointA][y] - triangle->v[pointC][y];
    F = triangle->v[pointA][z] - triangle->v[pointC][z];

    G = ray->directionVector[x];
    H = ray->directionVector[y];
    I = ray->directionVector[z];

    J = triangle->v[pointA][x] - ray->startPosition[x];
    K = triangle->v[pointA][y] - ray->startPosition[y];
    L = triangle->v[pointA][z] - ray->startPosition[z];

    M = A*(E*I - H*F) + B*(G*F - D*I) + C*(D*H - E*G);

    t = -(F*(A*K - J*B) + E*(J*C - A*L) + D*(B*L - K*C));
    t = t/M;

    gamma = (I*(A*K - J*B) + H*(J*C - A*L) + G*(B*L - K*C))/M;

    beta = (J*(E*I - H*F) + K*(G*F - D*I) + L*(D*H - E*G))/M;

    //checking for no hits
    if(t < 0){
        rayHit.time = -1;
        return rayHit;
    }
    if(gamma < 0 || gamma > 1){
        rayHit.time = -1;
        return rayHit;
    }

    float minusGamma = 1-gamma;

    if(beta < 0 || beta > minusGamma){
        rayHit.time = -1;
        return rayHit;
    }

    rayHit.time = t;
    rayHit.rayHitMat = triangle->mat;

    //find the intersection
    float hitPositions[3];
    float dt[3];
    scaleMultiply(rayHit.time, ray->directionVector, dt);
    addVectors(ray->startPosition, dt, hitPositions);

    rayHit.intersectLocation[0] = hitPositions[0];
    rayHit.intersectLocation[1] = hitPositions[1];
    rayHit.intersectLocation[2] = hitPositions[2];

    //find normal of surface
    float sideAB[3] = {0, 0,0};
    float sideCB[3] = {0, 0,0};
    float toNormalize[3] = {0, 0,0};


    subtractVector(triangle->v[pointA], triangle->v[pointB], sideAB);
    subtractVector(triangle->v[pointC], triangle->v[pointB], sideCB);


    //cross
    crossProduct(sideCB, sideAB, toNormalize);


    normalize(toNormalize);

    rayHit.normal[0] = toNormalize[0];
    rayHit.normal[1] = toNormalize[1];
    rayHit.normal[2] = toNormalize[2];



    return rayHit;
}

//edit number of spheres and tirables here
int numberOfSpheres = 3;
int numberOfTriangles = 5;

Sphere *Spheres;
Triangle  *Triangles;

//finding the closest intersects.
RayHit closestIntersects(Ray rayShot){
    RayHit rayHit = {.time = -1};
    int z, first = 0;
    for(z = 0; z < numberOfSpheres; z++){
        RayHit tempRayHit = sphereIntersection(&rayShot, &Spheres[z]);

        if(tempRayHit.time != -1) {
            if (tempRayHit.time > 0) {
                if (first == 0) {
                    rayHit = tempRayHit;
                    first++;
                } else if (tempRayHit.time < rayHit.time) {
                    rayHit = tempRayHit;
                }
            }
        }
    }

    for(z = 0; z < numberOfTriangles; z++){
        RayHit tempRayHit = triangleIntersect(&rayShot, &Triangles[z]);
        if(tempRayHit.time != -1) {
            if (tempRayHit.time > 0) {
                if (first == 0) {
                    rayHit = tempRayHit;
                    first++;
                } else if (tempRayHit.time < rayHit.time) {
                    rayHit = tempRayHit;
                }
            }
        }
    }

    return rayHit;
}

//this is a helper function for finding what color need to be changed in a pixel
int coordinate2Index(int row, int column, int color){
    int index = row * 3 * 512;
    index += column * 3;
    index += color;
    return index;
}

float light[3] = {3, 5, -15};


float getDiffuse();

int shadows(RayHit rayHit, const float lightVector[3]){
    //find the distance to the light
    float xDist = (rayHit.intersectLocation[0] - light[0]) * (rayHit.intersectLocation[0]- light[0]);
    float yDist = (rayHit.intersectLocation[1] - light[1]) * (rayHit.intersectLocation[1]- light[1]);
    float zDist = (rayHit.intersectLocation[2] - light[2]) * (rayHit.intersectLocation[2]- light[2]);

    float distanceToLight = sqrt(xDist + yDist + zDist);


    //shadow Ray
    Ray newRay = {
            .startPosition = {
                    rayHit.intersectLocation[0]+(lightVector[0]*.001),
                    rayHit.intersectLocation[1]+(lightVector[1]*.001),
                    rayHit.intersectLocation[2]+(lightVector[2]*.001),
            },
            .directionVector = {
                    lightVector[0],
                    lightVector[1],
                    lightVector[2]
            }
    };

    int isShadow = 0;

    RayHit hit = closestIntersects(newRay);

    if(hit.time > 0){
        if(hit.time < distanceToLight){
            isShadow = 1;
        }
    }


    return isShadow;

}



void reflect(RayHit rayHit, float *color, Ray incomingRay, float rayToLight[3]){

    Ray currentIncomingRay = incomingRay;
    RayHit currentHit = rayHit;

    int i = 0;
    for(i = 0; i < 11; i++){
        //getting the reflecting ray
        float reflectRay[3];
        float d_dot_n = dotProduct(currentIncomingRay.directionVector, currentHit.normal);
        float neg2_mul_d_dot_n = d_dot_n * -2;

        float neg2_mul_d_dot_n_mul_n[3];
        scaleMultiply(neg2_mul_d_dot_n, currentHit.normal, neg2_mul_d_dot_n_mul_n);
        addVectors(currentIncomingRay.directionVector,neg2_mul_d_dot_n_mul_n,reflectRay);
        normalize(reflectRay);

        Ray newRay = {
                .startPosition = {
                        currentHit.intersectLocation[0]+(reflectRay[0] *.001),
                        currentHit.intersectLocation[1]+(reflectRay[1] *.001),
                        currentHit.intersectLocation[2]+(reflectRay[2] *.001)
                },
                .directionVector = {
                        reflectRay[0],
                        reflectRay[1],
                        reflectRay[2]
                }
        };

        //check if the ray hits any other geometry
        RayHit nextHit = closestIntersects(newRay);

        //if it misses to hit a object then end the loop
        if(nextHit.time < 0){
            color[0] = 0;
            color[1] = 0;
            color[2] = 0;
            break;
        } else{
            //check if the object hit is reflective or not
            //if not reflective set the color to the color of the current hit diffuse
            if(nextHit.rayHitMat.reflective != 1) {

                float  diffuse = getDiffuse(nextHit);

                color[0] = nextHit.rayHitMat.color[0] * diffuse * 255;
                color[1] = nextHit.rayHitMat.color[1]  * diffuse *255;
                color[2] = nextHit.rayHitMat.color[2]  * diffuse * 255;
                break;
            } else {
                currentHit = nextHit;
                currentIncomingRay = newRay;
            }
        }


    }
}



float getDiffuse(RayHit rayHit){
    float rayToLight[3] = {0,0,0};
    subtractVector(light, rayHit.intersectLocation, rayToLight);
    normalize(rayToLight);


    int shadow = shadows(rayHit,rayToLight);
    float diffuse = dotProduct(rayHit.normal, rayToLight);


    if (shadow) {
        diffuse = .2;
    }

    if (diffuse < .2) {
        diffuse = .2;
    }



    return diffuse;
}

void getColor(RayHit rayHit, float *color, Ray incomingRay){


    float rayToLight[3] = {0,0,0};
    subtractVector(light, rayHit.intersectLocation, rayToLight);
    normalize(rayToLight);



    if(!rayHit.rayHitMat.reflective){
        float diffuse = getDiffuse(rayHit);
        color[0] = rayHit.rayHitMat.color[0]  * diffuse * 255;
        color[1] = rayHit.rayHitMat.color[1]  * diffuse * 255;
        color[2] = rayHit.rayHitMat.color[2] * diffuse * 255;
    } else {
        reflect(rayHit, color, incomingRay, rayToLight);
    }



}


int main() {

//edit objecst here (dont change above)
    //creating objects
    Spheres = (Sphere *)malloc(sizeof(Sphere) * numberOfSpheres);
    Triangles = (Triangle *)malloc(sizeof(Triangle) * numberOfTriangles);

    material reflective = { .color = {0,0,0}, .reflective = 1 };
    material blue = { .color = {0,0,1}, .reflective = 0 };
    material red =  { .color = {1,0,0}, .reflective = 0 };
    material white= { .color = {1,1,1}, .reflective = 0 };

    Spheres[0] = (Sphere) {.center = {0,0, -16}, .radius = 2, .mat = reflective};
    Spheres[1] = (Sphere) {.center = {3, -1, -14}, .radius = 1, .mat = reflective};
    Spheres[2] = (Sphere) {.center = {-3, -1, -14}, .radius = 1, .mat = red};

// back wall
    Triangles[0] = (Triangle) { .v = { { -8,-2,-20 }, {8,-2,-20}, {8,10,-20} }, .mat = blue };
    Triangles[1] = (Triangle) { .v = { { -8,-2,-20 }, {8,10,-20}, {-8,10,-20} }, .mat = blue };
// floor
    Triangles[2] = (Triangle) { .v = { { -8,-2,-20 }, {8,-2,-10}, {8,-2,-20}}, .mat = white };
    Triangles[3] = (Triangle) { .v = { { -8,-2,-20 }, {-8,-2,-10}, {8,-2,-10}}, .mat = white };
// right red triangle
    Triangles[4] = (Triangle) { .v = { { 8,-2,-20 }, {8,-2,-10}, {8,10,-20}}, .mat = red };
    numberOfTriangles = 5;
//end (dont change blow)


    int totalPixels = height * width * channels;

    unsigned char *img = malloc(totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        img[i] = 0;
    }

    int x, y;


    for(y = 0; y < height; y++){
        for(x = 0; x < width; x++){
            int index = coordinate2Index(y, x, 0);
            Ray rayShot;
            getRay(x, y, &rayShot);

            RayHit rayHit = {.time = -1};

            rayHit = closestIntersects(rayShot);


            if(rayHit.time > 0) {
                float color[3] = {0,0,0};
                getColor(rayHit, color, rayShot);
                img[index] = color[0];
                img[index+1] = color[1];
                img[index+2] = color[2];
            }

        }
    }




    stbi_write_png("reference.png", width, height, channels, img, width* channels);
    free(Spheres);
    free(Triangles);
    free(img);

    return 0;
}



