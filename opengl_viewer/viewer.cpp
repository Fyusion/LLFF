// OpenGL viewer based on
// https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/4.advanced_opengl/3.2.blending_sort/blending_sorted.cpp

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>

#include <utils/shader_m.h>
#include <utils/camera.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>


using std::vector;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void loadMPI2Tex(char const * path, unsigned int * tIDs, unsigned int width, unsigned int height, unsigned int depth);


// settings
unsigned int SCR_WIDTH = 1200;
unsigned int SCR_HEIGHT = 800;

// camera
// Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool do_simple=true;
int render_ind=3;
bool draw_guide_mesh=true;
bool center_only=false;
bool automove=false;
bool use_input_imgs=false;


float rx_global = 1., ry_global = 1., foc_global=1000., offset_global=0.;

glm::mat4 vecs2mat(glm::vec3 x_avg, glm::vec3 y_avg, glm::vec3 z_avg, glm::vec3 centroid) {

    glm::mat4 c2w_avg(
        x_avg[0], y_avg[0], z_avg[0], centroid[0],
        x_avg[1], y_avg[1], z_avg[1], centroid[1],
        x_avg[2], y_avg[2], z_avg[2], centroid[2],
               0,        0,        0,           1
        );
    return glm::transpose(c2w_avg);
}

glm::mat4 autoposition(glm::vec3 x, glm::vec3 y, glm::vec3 z, glm::vec3 c, float theta, float rx, float ry, float d) {
    // make pos using theta and r, focus at distance d away
    glm::vec3 focalpt = c - d * z;
    static float d_old;
    if (d != d_old)
        std::cout << "Focal depth " << d << std::endl;
    d_old = d;
    glm::vec3 pos = c + (rx * cosf(theta) * x + ry * sinf(theta) * y);
    glm::vec3 mz = glm::normalize(pos - focalpt);
    glm::vec3 mx = glm::normalize(glm::cross(y, mz));
    glm::vec3 my = glm::normalize(glm::cross(mz, mx));

    return vecs2mat(mx, my, mz, pos);

}


glm::vec3 mat2trans(glm::mat4 m) {
    return glm::vec3(m[3][0], m[3][1], m[3][2]);
}


unsigned int textured_quad;
unsigned int opengl_initTQ() {

    static const float planeVertices[] = {
        // positions         // texture Coords (swapped y coordinates because texture is flipped upside down)
        -.5f,  0.5f,  0.0f,  0.0f,  0.0f,
        -.5f, -0.5f,  0.0f,  0.0f,  1.0f,
         .5f, -0.5f,  0.0f,  1.0f,  1.0f,

        -.5f,  0.5f,  0.0f,  0.0f,  0.0f,
         .5f, -0.5f,  0.0f,  1.0f,  1.0f,
         .5f,  0.5f,  0.0f,  1.0f,  0.0f
    };

    unsigned int vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(0);
    return vao;
}

unsigned int opengl_initSQ() {
    unsigned int quad_VertexArrayID;
    glGenVertexArrays(1, &quad_VertexArrayID);
    glBindVertexArray(quad_VertexArrayID);

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.9f,
        1.0f,  -1.0f, 0.9f,
        -1.0f,  1.0f, 0.9f,
        -1.0f,  1.0f, 0.9f,
        1.0f,  -1.0f, 0.9f,
        1.0f,   1.0f, 0.9f,
    };

    GLuint quad_vertexbuffer;
    glGenBuffers(1, &quad_vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    return quad_VertexArrayID;
}


struct Framebuffer {
    GLuint fbn, renderTex;
    void init() {
        // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
        glGenFramebuffers(1, &fbn);
        glBindFramebuffer(GL_FRAMEBUFFER, fbn);

        // The texture we're going to render to
        glGenTextures(1, &renderTex);

        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture(GL_TEXTURE_2D, renderTex);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0,GL_RGBA, GL_UNSIGNED_BYTE, 0);

        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // The depth buffer
        GLuint depthrenderbuffer;
        glGenRenderbuffers(1, &depthrenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

        // Set "renderTex" as our colour attachement #0
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTex, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer is broken!!" << std::endl;

    }

    void bind() {

        glBindFramebuffer(GL_FRAMEBUFFER, fbn);
        // glViewport(0,0,SCR_WIDTH, SCR_HEIGHT);

        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
};

struct MPIData {
    unsigned int N_planes;
    unsigned int *mpiTex;
    int width, height;
    float focal, fovy;
    float *dvals;
    glm::mat4 c2w;
    glm::vec3 front;


    void load(const char* basedir, unsigned int w=0, unsigned int h=0, unsigned int N_planes=0) {
        this->N_planes = N_planes;

        char filestr[1024];
        sprintf(filestr, "%s/mpi.b", basedir);
        mpiTex = new unsigned int[N_planes];
        loadMPI2Tex(filestr, mpiTex, w, h, N_planes);
        width = w;
        height = h;


        sprintf(filestr, "%s/metadata.txt", basedir);
        std::ifstream file(filestr);
        std::string line;

        if (true) {
            float w, h, d;
            // CHANGE THIS
            // file >> h >> w >> focal;
            file >> h >> w >> d >> focal;
            float cam2world[16];
            for (int i = 0; i < 16; ++i) {
                if (i%4 != 3) 
                    file >> cam2world[i];
                else
                    cam2world[i] = i < 15 ? 0. : 1.;
            }
            c2w = glm::make_mat4(cam2world);
            glm::vec4 f = c2w * glm::vec4(0.,0.,-1.,0.);
            this->front = glm::vec3(f[0], f[1], f[2]);
            this->front = -glm::vec3(cam2world[2], cam2world[6], cam2world[10]);
            std::cout << "c2w matrix" << std::endl;
            for (int i = 0; i < 16; ++i) {
                std::cout << cam2world[i] << " ";
                if (i%4==3)
                    std::cout << std::endl;
            }

            float close_depth, inf_depth;
            file >> inf_depth >> close_depth;
            std::cout << "depths " << inf_depth << " " << close_depth << std::endl;
            dvals = new float[N_planes];
            for (int i = 0; i < N_planes; ++i) {
                float t = ((float)i) / (N_planes - 1.);
                dvals[i] = 1./((1.-t) / inf_depth + t / close_depth);
            }
        } else {
        
            file >> line;
            file >> focal;
            file >> line;
            for (int i = 0; i < N_planes; ++i) {
                file >> dvals[i];
            }
        }

        fovy = atan(height/focal*.5)*2.;

    }

    void draw(Shader& shader) {

        for (int i = 0; i < N_planes; ++i) {

            glBindTexture(GL_TEXTURE_2D, mpiTex[i]);

            glm::mat4 model = glm::mat4(c2w);
            model = glm::translate(model, glm::vec3( 0., 0., -dvals[i]));
            model = glm::scale(model, glm::vec3(dvals[i]*width/focal, dvals[i]*height/focal, 1.));
            shader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 6);

        }

    }

};


struct ViewMesh {
    vector<MPIData*> nodes;
    float scale;
    glm::vec3 x0, y0, z0, c0;
    MPIData* center_mpi;
    std::vector<std::pair<int, float>> dists;

    float theta, rx, ry, dtheta = 3.14159265 / 180. * 360./96.;
    glm::mat4 autoview() {
        glm::mat4 m = autoposition(x0, y0, z0, c0 - offset_global * z0, theta, rx*rx_global, ry*ry_global, foc_global);
        theta += dtheta;
        if (theta > 2. * 3.14159265) {
            theta -= 2. * 3.14159265;

        }
        return m;

    }

    void get_dists(glm::vec3 p) {
        dists.clear();
        
        glm::vec4 p4(p[0], p[1], p[2], 1.);
        for (int i = 0; i < nodes.size(); ++i) {
            glm::vec4 v = glm::column(nodes[i]->c2w, 3);
            float d = glm::length(v - p4);
            dists.push_back({i, d});
            // std::cout << i << ": " << d << std::endl;
        }
        std::sort(std::begin(dists), std::end(dists), 
            [](std::pair<int,float> a, std::pair<int,float> b) -> bool { return a.second < b.second; });

    }

    void loadvm(const char* basedir) {

        char filestr[1024];
        sprintf(filestr, "%s/metadata.txt", basedir);
        std::ifstream file(filestr);
        int N_nodes, N_tri;
        unsigned int width, height, N_planes;
        file >> N_nodes >> width >> height >> N_planes;
        std::cout << "loading " << N_nodes << " mpis, in " << N_planes << " depth planes" << std::endl;
        std::cout << width << " x " << height << std::endl;
        
        for (int i = 0; i < N_nodes; ++i) {
            char filestr[1024];
            sprintf(filestr, "%s/mpi%02d", basedir, i);
            std::cout << filestr << std::endl;
            MPIData *mpi = new MPIData();
            mpi->load(filestr, width, height, N_planes);
            nodes.push_back(mpi);
        }

        // Processing
        glm::mat4 mod = glm::inverse(nodes[0]->c2w);
        glm::vec4 centroid(0,0,0,0);
        glm::vec3 z_avg(0,0,0), up_avg(0,0,0);
        for (MPIData* node : nodes) {
            node->c2w = mod * node->c2w;
            centroid += node->c2w[3];
            z_avg += glm::vec3(node->c2w[2]);
            up_avg += glm::vec3(node->c2w[1]);
        }
        centroid /= nodes.size();
        z_avg = glm::normalize(z_avg);
        glm::vec3 x_avg = glm::normalize(glm::cross(up_avg, z_avg));
        glm::vec3 y_avg = glm::normalize(glm::cross(z_avg, x_avg));
        x0 = x_avg; y0 = y_avg; z0 = z_avg; c0 = centroid;
        float var = 0.;
        float maxl2 = 1e10;
        int ii = 0, mi = 0;
        for (MPIData* node : nodes) {
            glm::vec4 diff = node->c2w[3] - centroid;
            float l2 = glm::length2(diff);
            var += l2;
            if (l2 < maxl2) {
                maxl2 = l2;
                center_mpi = node;
                mi = ii;
            }
            ii += 1;
        }

        std::cout << "center MPI is " << mi << std::endl;
        // center_mpi = nodes[nodes.size()/2];
        var /= nodes.size();
        float r = sqrt(var);
        scale = 1./r * .5;
        rx = ry = r * .5;
        theta = 0.f;
        std::cout << "centroid of data " << glm::to_string(centroid) << std::endl;
        std::cout << "radius of data " << r << std::endl;


        foc_global = .25 * nodes[0]->dvals[0] + .75 * nodes[0]->dvals[nodes[0]->N_planes-1];

    }
};



struct GLMaster {
    ViewMesh viewmesh;

    Camera camera = Camera(glm::vec3(0.0f, 0.0f, 0.0f));

    Shader *shader, *shader_quad;

    unsigned int textured_quad, screen_quad;
    Framebuffer fbs[3];
    glm::mat4 projection;
    float fovy;

    char* framedata;
    ~GLMaster () {

    }

    void init(const char* viewmesh_file) {

        std::cout << "shader" << std::endl;
        shader = new Shader(
            "blending.vs", 
            "blending.fs");
        std::cout << "shader2" << std::endl;
        shader_quad = new Shader(
            "blending_quad.vs", 
            "blending_quad.fs");


        textured_quad = opengl_initTQ();
        screen_quad = opengl_initSQ();

        viewmesh.loadvm(viewmesh_file);

        // currtri = viewmesh.triangles[0];

        fovy = atan(viewmesh.nodes[0]->height/viewmesh.nodes[0]->focal*.5)*2.;
        std::cout << "fovy is " << fovy*180./3.1415 << " degs" << std::endl;
        projection = glm::perspective(fovy, (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100000.0f);

        camera.MovementSpeed = .2;
        camera.mousepan = true;

        camera.MouseSensitivityPan = -.01f * .2f / viewmesh.scale;
        camera.MouseSensitivityRot = -.05f;

        // if (camera.mousepan) {
        //     camera.MouseSensitivity = -.01f * .2f / viewmesh.scale;
        // } else {
        //     camera.MouseSensitivity = .05f;
        // }

        for (int i = 0; i < 3; ++i)
            fbs[i].init();


    }
    float deltaTime, lastFrame; 
    void render() {

        lastFrame = glfwGetTime();


        projection = glm::perspective(fovy, (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100000.0f);

        shader->use();
        shader->setInt("texture1", 0);
        shader->setMat4("projection", projection);


        glm::mat4 view = camera.GetViewMatrix();

        // view = glm::lookAt(glm::vec3(0,0,0), glm::vec3(0,0,-1), glm::vec3(0,1,0));


        glm::vec3 bco;
        if (automove) {
            view = viewmesh.autoview();
            // bco = choose_triangle(view[3]);
            viewmesh.get_dists(view[3]);
            view = glm::inverse(view);
        } else {
            // bco = choose_triangle(camera.Position);
            viewmesh.get_dists(camera.Position);
        }
        shader->setMat4("view", view);


        glm::mat4 c2w_curr = glm::inverse(view);

        float bcosum = 0;
        for (int i = 0; i < 3; ++i) {
            fbs[i].bind();
            glBindVertexArray(textured_quad);
            glActiveTexture(GL_TEXTURE0);
            if (center_only) {
                viewmesh.center_mpi->draw(*shader);
                bco[i] = 1.;
            }
            else {
                int ind = viewmesh.dists[i].first;
                // std::cout << "rank " << i << ", " << ind << std::endl;
                auto node = viewmesh.nodes[ind];
                if (use_input_imgs) {
                    shader->setMat4("view", glm::inverse(node->c2w)); 
                }
                node->draw(*shader);
                bco[i] = exp(-viewmesh.dists[i].second);
                bcosum += bco[i];
            }

        }

        for (int i = 0; i < 3; ++i) {
            bco[i] /= bcosum;
        }
        // std::cout << glm::to_string(bco) <<  std::endl;
        // budata.render2fb(*shader);

        // Render to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // glViewport(0,0,SCR_WIDTH, SCR_HEIGHT);

        // Clear the screen
        glClearColor(.2,.2,.2, 1.0f);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        shader_quad->use();
        glBindVertexArray(screen_quad);
        // glEnableVertexAttribArray(0);


        if (render_ind < 0) {
            bco = glm::vec3(1./3, 1./3, 1./3);
        } else if (render_ind == 3) {

            for (int i = 0; i < 3; ++i) {
                if (bco[i] > 1.) bco[i] = 1.;
                if (bco[i] < 0.) bco[i] = 0.;
            }
        } else {
            bco = glm::vec3(0,0,0);
            bco[render_ind] = 1.;
        }

        shader_quad->setVec3("bco", bco);

        // glBindTexture(GL_TEXTURE_2D, budata[render_ind].renderTex);

        for (int i = 0; i < 3; ++i) {
            char texstr[64];
            sprintf(texstr, "texture%d", i+1);
            shader_quad->setInt(texstr, i);
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, fbs[i].renderTex);
        }

        glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles


        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // std::cout << "Delta T " << 1./deltaTime << std::endl;
    }

};
GLMaster glmaster;
Camera& camera = glmaster.camera;

int main(int argc, char* argv[])
{
    char filestr[1024];
    sprintf(filestr, "%s/metadata.txt", argv[1]);
    std::ifstream file(filestr);
    int N_nodes, N_tri;
    unsigned int width, height, depth;
    file >> N_nodes >> width >> height >> depth;
    std::cout << "RES " << width << " " << height << " " << depth << std::endl;
    SCR_WIDTH = width*2;
    SCR_HEIGHT = height*2;


    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    int scr_w=SCR_WIDTH, scr_h=SCR_HEIGHT;
    GLFWwindow* window = glfwCreateWindow(scr_w, scr_h, "Local Light Field Fusion OpenGL Viewer", NULL, NULL);

    int fb_w, fb_h;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    if (fb_w > scr_w) {
        std::cout << "Adjusting window size for high DPI display" << std::endl;
        scr_w /= 2;
        scr_h /= 2;
        glfwSetWindowSize(window, scr_w, scr_h);
    }

    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    std::cout << "Framebuffer size " << fb_w << " " << fb_h << " vs " << scr_w << " " << scr_h << std::endl;

    // glfw window creation
    // --------------------
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetMouseButtonCallback(window, mouse_button_callback);


    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    std::cout << "initializing viewmesh with " << argv[1] << std::endl;
    glmaster.init(argv[1]);



    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // std::cout << "Mega T " << 1./deltaTime << std::endl;

        // input
        // -----
        processInput(window);

        glmaster.render();


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    // glDeleteVertexArrays(1, &cubeVAO);
    // glDeleteVertexArrays(1, &planeVAO);
    // glDeleteBuffers(1, &cubeVBO);
    // glDeleteBuffers(1, &planeVBO);

    glfwTerminate();
    return 0;
}


bool keypress_array[1024];
bool keypress_tracker(GLFWwindow *window, int glfw_code, bool& val) {
    static bool first=true;
    bool retval = false;
    if (first) {
        first = false;
        for (int i = 0; i < 1024; ++i)
            keypress_array[i] = false;
    }
    if (glfwGetKey(window, glfw_code) == GLFW_PRESS) {
        if (!keypress_array[glfw_code]) {
            keypress_array[glfw_code] = true;
            val = !val;
            std::cout << "keypress " << 'A' + glfw_code - GLFW_KEY_A << std::endl;
            retval = true;
        }
    } else if (glfwGetKey(window, glfw_code) == GLFW_RELEASE) {
        keypress_array[glfw_code] = false;
    } 
    return retval;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);


    keypress_tracker(window, GLFW_KEY_Z, do_simple);
    keypress_tracker(window, GLFW_KEY_X, center_only);
    keypress_tracker(window, GLFW_KEY_L, automove);
    keypress_tracker(window, GLFW_KEY_I, use_input_imgs);
    
    // if (keypress_tracker(window, GLFW_KEY_V, record_video))
    //     glmaster.viewmesh.theta = 0.;



    for (int i = -1; i < 4; ++i) {

        if (glfwGetKey(window, GLFW_KEY_1+i) == GLFW_PRESS)
            render_ind=i;
    }


    float dr = 1.05;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        rx_global *= dr;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        rx_global /= dr;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        ry_global *= dr;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        ry_global /= dr;

    if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
        foc_global *= dr;
    if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
        foc_global /= dr;

    float dx = 1.;
    bool og_mod = false;
    if (glfwGetKey(window, GLFW_KEY_SEMICOLON) == GLFW_PRESS) {
        offset_global += dx;
        og_mod = true;
    }
    if (glfwGetKey(window, GLFW_KEY_APOSTROPHE) == GLFW_PRESS) {
        offset_global -= dx;
        og_mod = true;
    }
    if (og_mod) {
        std::cout << "offset_global " << offset_global << std::endl;
    }





    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            std::cout << "PRESS" << std::endl;
            camera.mousepan = false;
        } else {
            std::cout << "RELEASE" << std::endl;
            camera.mousepan = true;

        }
    }
}


// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // std::cout << "SCROLL " << yoffset << std::endl;
    glmaster.fovy *= exp(yoffset*.01);
    // camera.ProcessMouseScroll(yoffset);
}


void loadMPI2Tex(char const * path, unsigned int * tIDs, unsigned int width, unsigned int height, unsigned int depth)
{
    glGenTextures(depth, tIDs);
    unsigned int IMGSIZE = width * height * 4;
    unsigned int BUFFERSIZE = IMGSIZE * depth;
    char *data = new char [BUFFERSIZE];

    FILE * filp = fopen(path, "rb"); 
    int bytes_read = fread(data, sizeof(char), BUFFERSIZE, filp);
    if (bytes_read == BUFFERSIZE)
    {
        for (int i = 0; i < depth; ++i) {
            GLenum format = GL_RGBA;

            glBindTexture(GL_TEXTURE_2D, tIDs[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data + IMGSIZE * i);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        }
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
    }

    delete [] data;
}