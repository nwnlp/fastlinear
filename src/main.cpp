

#include <application.hpp>


int main(int argc, char **argv){
    Application app;
    app.Init("logistic regression");
    app.Train();
    return 0;
}