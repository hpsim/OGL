#!/bin/sh

# Sets a solver 
setSolver(){
    echo "Setting solver"
    baseFolder=$PWD
    cd Tests/$1
    sed "s/p{}/$2/g" system/fvSolutionTMP > system/fvSolution
    cd $baseFolder
}

# Derive a case from base case
deriveCase(){
    baseFolder=$PWD
    echo "Deriving case " $1 $2
    cd Tests
    mkdir -p $1
    cp -r pitzDailyBase $1/$2
    cd $baseFolder
}

# Copy Pitzdaily base case
generateBaseCase() {
    FILE=Tests/pitzDailyBase/system/controlDict
    baseFolder=$PWD
    if [ ! -f "$FILE" ]; then
        echo "Preparing base case"
        mkdir -p Tests
        cp -r $FOAM_TUTORIALS/incompressible/pisoFoam/LES/pitzDaily/ Tests/pitzDailyBase

        echo "Meshing base case"
        cd Tests/pitzDailyBase && blockMesh > blockMesh.log

        echo "add OGL.so"
        echo 'libs ("libOGL.so");' >> system/controlDict

        touch system/controlDict.foam

        # run only for 100 time steps
        sed -i 's/endTime[ ]*0.1/endTime 1e-3/g' system/controlDict

        # clear p solver settings
        sed -z 's/p\n[ ]*[{][^}]*[}]/p{}/g' system/fvSolution > system/fvSolutionTMP
    fi
    cd $baseFolder
}

runSolver() {
    baseFolder=$PWD
    cd Tests/$1
    echo "Starting solver"
    pisoFoam | tee log | grep "Time ="
    cd $baseFolder
}

basePOFCG(){
    solver="OFCG"
    deriveCase "p" $solver
    setSolver  "p/$solver" "p{solver PCG;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000;}"
    runSolver "p/$solver"
}


basePGKOCG(){
    solver="GKOCG"
    deriveCase "p" "$solver"
    setSolver  "p/$solver" "p{solver $solver;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000;}"
    runSolver "p/$solver"
}

basePGKOCGOMP(){
    solver="GKOCGOMP"
    deriveCase "p" "$solver"
    setSolver  "p/$solver" "p{solver GKOCG;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000; executor omp;}"
    runSolver "p/$solver"
}


# run symmetric solver

generateBaseCase

basePGKOCGOMP
basePOFCG
basePGKOCG


