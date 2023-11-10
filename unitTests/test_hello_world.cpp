#include "gtest/gtest.h"
#include "HostMatrix/HostMatrix.H"

TEST(HelloWord, check_return_value){
    int my_argc {1};
    string str = "icoFoam";
    std::vector<char> cstr(str.c_str(), str.c_str() + str.size() + 1);
    char* argv = cstr.data();
    char** argvp = &argv;

    Foam::argList args(my_argc, argvp);
    Foam::Time runTime(Foam::Time::controlDictName, argv);

    Foam::objectRegistry objectReg{runTime};

    EXPECT_EQ(1, 1);
}
